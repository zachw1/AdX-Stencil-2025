from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.local_games.adx_arena import AdXGameSimulator
from agt_server.agents.utils.adx.structures import Bid, Campaign, BidBundle, MarketSegment 
from typing import Set, Dict
import math

class MyNDaysNCampaignsAgent(NDaysNCampaignsAgent):
    """
    Advanced AdX agent based on the TAC AdX specification.
    
    Key Strategies:
    1. Marginal Value Bidding: Bids based on derivative of effective reach (sigmoidal)
    2. Quality Score Optimization: Prioritizes campaign completion for reputation
    3. Expected Profit Calculation: Only accepts profitable campaigns
    4. Second-Price Truthful Bidding: Bids true marginal value in auctions
    """

    def __init__(self):
        super().__init__()
        self.name = "EnhancedBidder"
        
        # Campaign management: prioritize completion over quantity

        # Note: Quality score affects free campaign probability (p = min(1, Q))
        self.max_active_campaigns = 3
        
        # Effective reach function constants from spec
        self.a = 4.08577
        self.b = 3.08577
        
        # Quality score update: Q_new = (1-α)*Q_old + α*avg_effective_reach
        # α = 0.5 from config (line 88 in adx_arena.py)
        self.quality_alpha = 0.5
        
        # Segment size estimates from config (lines 55-62 in adx_arena.py)
        # These are the EXACT population sizes used by the simulator
        self.segment_sizes = {
            'Male_Young_LowIncome': 1836,
            'Male_Young_HighIncome': 517,
            'Male_Old_LowIncome': 1795,
            'Male_Old_HighIncome': 808,
            'Female_Young_LowIncome': 1980,
            'Female_Young_HighIncome': 256,
            'Female_Old_LowIncome': 2401,
            'Female_Old_HighIncome': 407,
        }
        
        # Track actual costs for learning (from implementation observations)
        self.observed_costs = []

    def on_new_game(self) -> None:
        """Initialize game-specific state."""
        self.observed_costs = []

    def get_ad_bids(self) -> Set[BidBundle]:
        """
        Bid on ad impressions using MARGINAL VALUE bidding.
        
        Key insights from implementation (adx_arena.py):
        1. Second-price auction: winner pays 2nd highest bid (line 187)
        2. 10,000 users arrive per day from atomic segments (line 364)
        3. Bids checked against BOTH per-bid limit AND bundle limit (lines 189-190)
        4. Only impressions where target_segment ⊆ user_segment count (line 203)
        
        Effective reach ρ(C) is SIGMOIDAL:
        - First impressions: LOW marginal value
        - Middle impressions: HIGH marginal value  
        - Near-completion: HIGHEST marginal value
        - Beyond reach R: diminishing returns (asymptote 1.38442)
        
        Strategy: Bid true marginal value = dρ/dx × Budget (dominant in 2nd-price)
        """
        bundles = set()
        current_day = self.get_current_day()

        for campaign in self.get_active_campaigns():
            impressions_won = self.get_cumulative_reach(campaign) or 0
            cost_so_far = self.get_cumulative_cost(campaign) or 0.0
            
            # Calculate remaining budget
            remaining_budget = max(0.0, campaign.budget - cost_so_far)
            if remaining_budget <= 0.0:
                continue
            
            # Stop bidding at reach (diminishing returns kick in)
            if impressions_won >= campaign.reach:
                continue
            
            # Calculate MARGINAL VALUE of next impression
            # Marginal value = (dρ/dx) × Budget
            marginal_value = self._marginal_effective_reach(
                impressions_won, 
                campaign.reach
            ) * campaign.budget
            
            # Adjust for schedule urgency
            days_left = max(1, campaign.end_day - current_day + 1)
            total_duration = max(1, campaign.end_day - campaign.start_day + 1)
            days_elapsed = max(0, current_day - campaign.start_day)
            
            # Progress tracking
            completion_frac = impressions_won / float(campaign.reach)
            time_frac = days_elapsed / float(total_duration)
            
            # Behind schedule? Bid more aggressively
            if time_frac > 0:
                progress_ratio = completion_frac / time_frac
                if progress_ratio < 0.7:
                    # Significantly behind - urgent
                    urgency_factor = 1.4
                elif progress_ratio < 0.85:
                    # Somewhat behind
                    urgency_factor = 1.2
                elif progress_ratio > 1.3:
                    # Well ahead - can be conservative
                    urgency_factor = 0.85
                else:
                    # On track
                    urgency_factor = 1.0
            else:
                urgency_factor = 1.0
            
            # Final day: bid aggressively to complete
            if days_left == 1:
                urgency_factor *= 1.3
            
            # Calculate bid per item
            bid_per_item = marginal_value * urgency_factor
            
            # Safety bounds (market rarely exceeds these values)
            bid_per_item = max(0.05, min(2.0, bid_per_item))
            
            # Calculate daily spending limit for budget pacing
            remaining_reach = campaign.reach - impressions_won
            if days_left > 1:
                # Spread remaining budget across remaining days
                # Allow 1.8x daily average to capture opportunities
                target_daily_budget = remaining_budget / float(days_left)
                bid_limit = min(remaining_budget, target_daily_budget * 1.8)
            else:
                # Final day: use all remaining budget
                bid_limit = remaining_budget
            
            # Ensure we can afford at least some impressions
            bid_limit = max(bid_limit, bid_per_item * 10)
            bid_limit = min(bid_limit, remaining_budget)
            
            # CRITICAL: bid_per_item MUST be <= bid_limit (assertion in Bid constructor)
            bid_per_item = min(bid_per_item, bid_limit)
            
            # Ensure bid_per_item is strictly positive for valid bid
            if bid_per_item <= 0 or bid_limit <= 0:
                continue
            
            # Create bid bundle
            bid = Bid(
                bidder=self,
                auction_item=campaign.target_segment,
                bid_per_item=bid_per_item,
                bid_limit=bid_limit
            )
            
            bundle = BidBundle(
                campaign_id=campaign.uid,
                limit=bid_limit,
                bid_entries={bid}
            )
            bundles.add(bundle)
        
        return bundles
    
    def _marginal_effective_reach(self, x: int, R: int) -> float:
        """
        Calculate derivative of effective reach function dρ/dx at current impressions.
        
        From spec: ρ(C) = (2/a) × [arctan(a×(x/R) - b) - arctan(-b)]
        
        Derivative: dρ/dx = (2/a) × 1/(1 + (a×(x/R) - b)²) × (a/R)
                          = 2/(R × (1 + (a×(x/R) - b)²))
        
        This gives us the marginal value of one additional impression.
        """
        if R <= 0:
            return 0.0
        
        ratio = x / float(R)
        term = self.a * ratio - self.b
        denominator = R * (1 + term * term)
        
        if denominator <= 0:
            return 0.0
        
        return 2.0 / denominator

    def get_campaign_bids(self, campaigns_for_auction: Set[Campaign]) -> Dict[Campaign, float]:
        """
        Bid in campaign REVERSE auction using expected profit analysis.
        
        Key insights from implementation (adx_arena.py):
        1. Reverse auction: lowest effective_bid wins (line 233)
        2. effective_bid = agent_bid / quality_score (line 225)
           → Higher Q gives HUGE advantage in winning
        3. Winner pays: 2nd_lowest_bid × winner_Q (line 244)
           → High Q means we bid high but pay competitive price
        4. Only bidder: budget = (reach/avg_low_3_Q) × winner_Q (lines 234-241)
           → Can be very profitable if others have low Q
        5. Free campaigns: p = min(1, Q), budget = reach (lines 403-406)
        
        Strategy: 
        - Calculate expected profit for each campaign
        - Bid maximum willingness to pay (truthful in 2nd-price)
        - Quality score is CRITICAL: complete campaigns → high Q → more wins + free campaigns
        """
        bids = {}
        
        active_campaigns = list(self.get_active_campaigns())
        active_count = len(active_campaigns)
        
        # Don't bid if at capacity (focus on completing current campaigns)
        if active_count >= self.max_active_campaigns:
            return bids
        
        max_new_campaigns = self.max_active_campaigns - active_count
        quality_score = self.get_quality_score() or 1.0
        
        candidates = []
        
        for campaign in campaigns_for_auction:
            if campaign is None:
                continue
            
            duration = campaign.end_day - campaign.start_day + 1
            if duration <= 0:
                continue
            
            # Estimate difficulty
            est_users_per_day = self._estimate_segment_size(campaign.target_segment)
            total_est_imps = est_users_per_day * duration
            
            if total_est_imps > 0:
                fraction_needed = campaign.reach / total_est_imps
            else:
                fraction_needed = float("inf")
            
            # HARD FILTERS: Skip impossible/unprofitable campaigns
            
            # Too difficult (need >90% of available impressions)
            if fraction_needed > 0.9:
                continue
            
            # 1-day campaigns with high difficulty are risky
            if duration == 1 and fraction_needed > 0.7:
                continue
            
            # Calculate EXPECTED PROFIT
            expected_profit, expected_cost, expected_reach_fraction = self._estimate_campaign_profit(
                campaign, 
                fraction_needed,
                duration
            )
            
            # Only consider campaigns with positive expected profit
            if expected_profit <= 0:
                continue
            
            # Check for segment overlap (reduces available impressions)
            same_segment_active = sum(
                1 for c in active_campaigns
                if c.target_segment.name == campaign.target_segment.name
            )
            
            # Penalize if we already have campaigns on this segment
            if same_segment_active >= 2:
                continue  # Too much overlap, skip
            elif same_segment_active == 1:
                expected_profit *= 0.6  # Significant penalty
            
            # Calculate our maximum willingness to pay (for reverse auction)
            # Use estimated budget (campaign.budget is None during auction!)
            estimated_budget = campaign.reach * 0.45
            expected_revenue = expected_reach_fraction * estimated_budget
            min_profit_margin = expected_revenue * 0.15  # Want at least 15% margin
            max_willing_to_pay = expected_revenue - min_profit_margin
            
            # In reverse auction, bid our maximum willingness to pay
            # (truthful bidding in 2nd-price auction)
            bid_value = max_willing_to_pay
            
            # Adjust for difficulty: bid lower on harder campaigns (more risk)
            if fraction_needed > 0.7:
                bid_value *= 0.85
            elif fraction_needed < 0.3:
                bid_value *= 1.1  # Easy campaign, can bid higher
            
            # Adjust for quality score
            # Higher Q gives us advantage (effective_bid = bid/Q)
            # So with higher Q, we can afford to bid slightly higher
            if quality_score > 1.15:
                bid_value *= 1.05
            elif quality_score < 0.85:
                bid_value *= 0.95
            
            # Ensure bid is in valid range [0.1×reach, 1.0×reach]
            bid_value = max(campaign.reach * 0.1, min(campaign.reach * 1.0, bid_value))
            
            candidates.append({
                "campaign": campaign,
                "expected_profit": expected_profit,
                "bid_value": bid_value,
                "fraction_needed": fraction_needed,
                "duration": duration,
            })
        
        if not candidates:
            return bids
        
        # Sort by expected profit (most profitable first)
        candidates.sort(key=lambda c: c["expected_profit"], reverse=True)
        
        # Bid on top campaigns up to our capacity
        campaigns_bid_count = 0
        for cand in candidates:
            if campaigns_bid_count >= max_new_campaigns:
                break
            
            campaign = cand["campaign"]
            bid_value = cand["bid_value"]
            
            if self.is_valid_campaign_bid(campaign, bid_value):
                bids[campaign] = bid_value
                campaigns_bid_count += 1
        
        return bids
    
    def _estimate_campaign_profit(self, campaign: Campaign, fraction_needed: float, duration: int) -> tuple:
        """
        Estimate expected profit for a campaign.
        
        NOTE: campaign.budget is None during auctions! We must estimate it.
        
        Returns: (expected_profit, expected_cost, expected_reach_fraction)
        """
        # CRITICAL: Campaigns don't have budgets until after auction
        # From adx_arena.py lines 234-244:
        # - Multiple bidders: budget = 2nd_lowest_bid × winner_Q
        # - Single bidder: budget = (reach / avg_low_3_Q) × winner_Q
        # Estimate we'll pay around 0.4-0.5× reach
        estimated_budget = campaign.reach * 0.45
        
        # Estimate cost per impression based on market competition
        # Harder campaigns (higher fraction_needed) have higher competition
        if fraction_needed < 0.3:
            estimated_cpi = 0.008  # Easy, low competition
        elif fraction_needed < 0.5:
            estimated_cpi = 0.012  # Moderate
        elif fraction_needed < 0.7:
            estimated_cpi = 0.018  # Hard
        else:
            estimated_cpi = 0.025  # Very hard
        
        # Estimate how much of the campaign we'll actually complete
        # Based on fraction_needed and duration
        if fraction_needed < 0.4:
            expected_completion = 0.95  # Easy to complete
        elif fraction_needed < 0.6:
            expected_completion = 0.85
        elif fraction_needed < 0.8:
            expected_completion = 0.70
        else:
            expected_completion = 0.55  # Hard to complete
        
        # Longer campaigns are easier to complete
        if duration >= 3:
            expected_completion = min(1.0, expected_completion * 1.15)
        
        # Calculate expected impressions and cost
        expected_impressions = campaign.reach * expected_completion
        expected_cost = expected_impressions * estimated_cpi
        
        # Calculate expected effective reach
        expected_reach_fraction = self.effective_reach(
            int(expected_impressions), 
            campaign.reach
        )
        
        # Calculate expected profit using ESTIMATED budget
        expected_revenue = expected_reach_fraction * estimated_budget
        expected_profit = expected_revenue - expected_cost
        
        return expected_profit, expected_cost, expected_reach_fraction
    
    def _estimate_segment_size(self, segment: MarketSegment) -> float:
        """
        Estimate expected users per day for a segment.
        
        For atomic segments (e.g., Male_Young_LowIncome), use exact values.
        For broad segments (e.g., Male_Young, Male), sum all matching atomic segments.
        """
        name = segment.name
        
        # Exact atomic segment
        if name in self.segment_sizes:
            return float(self.segment_sizes[name])
        
        # Broad segment: sum matching atomic segments
        seg_tokens = name.split('_')
        total = 0
        
        for atomic_name, size in self.segment_sizes.items():
            atomic_tokens = atomic_name.split('_')
            # Check if all tokens of broad segment appear in atomic segment
            if all(tok in atomic_tokens for tok in seg_tokens):
                total += size
        
        return float(max(total, 1))

if __name__ == "__main__":
    # Here's an opportunity to test offline against some TA agents. Just run this file to do so.
    test_agents = [MyNDaysNCampaignsAgent()] + [Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)]

    # Don't change this. Adapt initialization to your environment
    simulator = AdXGameSimulator()
    simulator.run_simulation(agents=test_agents, num_simulations=1)