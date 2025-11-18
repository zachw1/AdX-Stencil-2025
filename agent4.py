from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.local_games.adx_arena import AdXGameSimulator
from agt_server.agents.utils.adx.structures import Bid, Campaign, BidBundle, MarketSegment 
from typing import Set, Dict
import math



class MyNDaysNCampaignsAgent(NDaysNCampaignsAgent):
    """
    Aggressive AdX agent inspired by Big Bidder's winning strategy.
    
    Key Strategies (learned from testing):
    1. AGGRESSIVE Campaign Bidding: Bid maximum (1.0× reach) to actually win campaigns
    2. More Campaigns: 5 concurrent campaigns for more opportunities
    3. Full Budget Usage: Use all remaining budget per day (no pacing)
    4. Smart Filtering: Skip 1-day campaigns and difficult campaigns (>50% segment)
    5. Average Value Ad Bidding: Bid average value over remaining impressions
    
    Why Aggressive Wins:
    - Conservative bidding → Lose auctions → Few campaigns → Failures tank Q → Death spiral
    - Aggressive bidding → Win campaigns → More chances to succeed → Maintain/boost Q
    """

    def __init__(self):
        super().__init__()
        self.name = "AggressiveBidder"
        
        # Campaign management: Match Big Bidder's aggressive strategy
        # More campaigns = more chances to succeed and recover from failures
        # Note: Quality score affects free campaign probability (p = min(1, Q))
        self.max_active_campaigns = 5  # Increased from 3
        
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
        
        # Logging for comparison
        self.campaign_history = {}
        self.daily_quality_scores = []
        self.ad_bid_history = {}

    def on_new_game(self) -> None:
        """Initialize game-specific state."""
        self.observed_costs = []
        self.campaign_history.clear()
        self.daily_quality_scores.clear()
        self.ad_bid_history.clear()

    def get_ad_bids(self) -> Set[BidBundle]:
        """
        Bid on ad impressions using AVERAGE VALUE bidding.
        
        Key insights from implementation (adx_arena.py):
        1. Second-price auction: winner pays 2nd highest bid (line 187)
        2. 10,000 users arrive per day from atomic segments (line 364)
        3. Bids checked against BOTH per-bid limit AND bundle limit (lines 189-190)
        4. Only impressions where target_segment ⊆ user_segment count (line 203)
        5. We bid for MANY impressions per day, not just one!
        
        Effective reach ρ(C) is SIGMOIDAL:
        - First impressions: LOW value
        - Middle impressions: HIGH value  
        - Near-completion: HIGHEST value
        - Beyond reach R: diminishing returns (asymptote 1.38442)
        
        Strategy: Bid average value per impression = Δρ/Δx × Budget / remaining_reach
        This accounts for winning multiple impressions with a single bid price.
        """
        bundles = set()
        current_day = self.get_current_day()
        
        # Track quality score
        quality_score = self.get_quality_score() or 1.0
        if not self.daily_quality_scores or self.daily_quality_scores[-1][0] != current_day:
            self.daily_quality_scores.append((current_day, quality_score))

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
            
            # Calculate AVERAGE VALUE per impression over remaining reach
            # We're bidding for many impressions, not just the next one!
            # Average value = Δρ/Δx over the remaining impressions
            remaining_reach = campaign.reach - impressions_won
            
            # Calculate value we'd gain from completing the campaign
            current_rho = self.effective_reach(impressions_won, campaign.reach)
            target_rho = self.effective_reach(campaign.reach, campaign.reach)  # = 1.0
            delta_rho = target_rho - current_rho
            
            # Average value per impression = (total remaining value) / (remaining impressions)
            avg_value_per_impression = (delta_rho * campaign.budget) / remaining_reach if remaining_reach > 0 else 0

            
            
            # Calculate bid per item using average value
            bid_per_item = avg_value_per_impression 
            
            # Use FULL remaining budget per day (like Big Bidder)
            # Pacing was too conservative - prevented winning enough impressions
            bid_limit = remaining_budget
            
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
            
            # Track campaign history and ad bids for logging
            cid = campaign.uid
            if cid not in self.campaign_history:
                self.campaign_history[cid] = {
                    "uid": cid,
                    "segment": campaign.target_segment.name,
                    "reach": campaign.reach,
                    "budget": campaign.budget,
                    "start_day": campaign.start_day,
                    "end_day": campaign.end_day,
                    "final_impressions": impressions_won,
                    "final_cost": cost_so_far,
                }
            else:
                self.campaign_history[cid]["final_impressions"] = impressions_won
                self.campaign_history[cid]["final_cost"] = cost_so_far
            
            if cid not in self.ad_bid_history:
                self.ad_bid_history[cid] = []
            self.ad_bid_history[cid].append({
                "day": current_day,
                "bid_per_item": float(bid_per_item),
                "bid_limit": float(bid_limit),
                "remaining_reach": int(remaining_reach),
                "remaining_budget": float(remaining_budget),
                "avg_value": float(avg_value_per_impression),
                "market_segment": campaign.target_segment.name,
            })
        
        return bundles

    def get_campaign_bids(self, campaigns_for_auction: Set[Campaign]) -> Dict[Campaign, float]:
        """
        Bid in campaign REVERSE auction - AGGRESSIVE STRATEGY.
        
        Key lessons from Big Bidder's success:
        1. BID MAXIMUM (1.0× reach) - conservative bidding = losing auctions
        2. More campaigns = more chances to succeed and recover from failures
        3. Skip 1-day campaigns - too risky, no margin for error
        4. Skip difficult campaigns (>50% of segment impressions needed)
        5. Avoid segment overlap with active campaigns
        
        Quality score is CRITICAL:
        - Completing campaigns → high Q → more wins + free campaigns
        - Failing campaigns → low Q → death spiral
        - Solution: Be aggressive, win campaigns, complete them
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
            
            # Skip ALL 1-day campaigns - too risky!
            # Both our failed campaigns were 1-day → 0 impressions → Q death spiral
            if duration == 1:
                continue
            
            # Too difficult (need >50% of available impressions)
            if fraction_needed > 0.5:
                continue
            
            # Check for segment overlap (reduces available impressions)
            same_segment_active = sum(
                1 for c in active_campaigns
                if c.target_segment.name == campaign.target_segment.name
            )
            
            # Skip if too much overlap
            if same_segment_active >= 1:
                continue  # Avoid competing with ourselves
            
            # AGGRESSIVE BIDDING STRATEGY (like Big Bidder)
            # Conservative bidding = losing auctions = no campaigns = Q death spiral
            # Solution: Bid at maximum allowed (1.0× reach)
            bid_value = campaign.reach  # Maximum allowed bid
            
            candidates.append({
                "campaign": campaign,
                "bid_value": bid_value,
                "fraction_needed": fraction_needed,
                "duration": duration,
            })
        
        if not candidates:
            return bids
        
        # Sort by difficulty (easiest campaigns first)
        # Easier campaigns = higher chance of completion = better Q
        candidates.sort(key=lambda c: c["fraction_needed"])
        
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
    
    def print_debug_summary(self):
        """Print post-game summary for comparison with other agents."""
        print("\n" + "=" * 100)
        print(f"POST-GAME SUMMARY for {self.name}")
        print("=" * 100)

        # Quality scores
        if self.daily_quality_scores:
            print("\nQuality score by day:")
            for day, q in self.daily_quality_scores:
                print(f"  Day {day}: Q = {q:.4f}")
        else:
            print("\nNo quality score history recorded.")

        # Campaign outcomes
        if self.campaign_history:
            print("\nCampaign outcomes:")
            for cid, info in sorted(self.campaign_history.items()):
                R = info["reach"]
                B = info["budget"]
                x = info["final_impressions"]
                k = info["final_cost"]
                rho = self.effective_reach(x, R) if R > 0 else 0.0
                approx_profit = rho * B - k
                completion_pct = (x / R * 100) if R > 0 else 0

                print(
                    f"  Campaign {cid} [{info['segment']}]: "
                    f"reach={R}, budget={B:.2f}, "
                    f"impressions={x} ({completion_pct:.1f}%), cost={k:.2f}, "
                    f"rho={rho:.3f}, profit={approx_profit:.2f}, "
                    f"days={info['start_day']}–{info['end_day']}"
                )
        else:
            print("\nNo campaign history recorded.")

        # Ad bid details
        print(f"\n{'=' * 100}")
        print("AD BIDDING DETAILS")
        print(f"{'=' * 100}")
        
        for cid, bids in self.ad_bid_history.items():
            if cid in self.campaign_history:
                segment = self.campaign_history[cid]["segment"]
                print(f"\nCampaign {cid} ({segment}):")
                for entry in bids:
                    day = entry["day"]
                    bid_per_item = entry["bid_per_item"]
                    bid_limit = entry["bid_limit"]
                    remaining_reach = entry["remaining_reach"]
                    avg_value = entry.get("avg_value", 0)
                    urgency = entry.get("urgency_factor", 1.0)

                    print(
                        f"  Day {day}: bid=${bid_per_item:.3f}, "
                        f"limit=${bid_limit:.2f}, "
                        f"need={remaining_reach} imps, "
                        f"avg_val=${avg_value:.3f}, urgency={urgency:.2f}x"
                    )

        print("=" * 100 + "\n")

if __name__ == "__main__":
    # Here's an opportunity to test offline against some TA agents. Just run this file to do so.
    test_agents = [MyNDaysNCampaignsAgent(),] + [Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)]

    # Don't change this. Adapt initialization to your environment
    simulator = AdXGameSimulator()
    simulator.run_simulation(agents=test_agents, num_simulations=1)