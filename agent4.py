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

    def __init__(self, name = "clash royale"):
        super().__init__()
        
        self.name = name
        self.max_active_campaigns = 5  # Increased from 3
        
        # Effective reach function constants from spec
        self.a = 4.08577
        self.b = 3.08577
        
        
        self.segment_sizes = {
            'Male_Young_LowIncome': 1836,
            'Male_Young_HighIncome': 517,
            'Male_Old_LowIncome': 1795,
            'Male_Old_HighIncome': 808,
            'Female_Young_LowIncome': 1980,
            'Female_Young_HighIncome': 256,
            'Female_Old_LowIncome': 2401,
            'Female_Old_HighIncome': 407,
            
            # 2-feature segments 
            'Male_Young': 2353,      # 1836 + 517
            'Male_Old': 2603,        # 1795 + 808
            'Male_LowIncome': 3631,  # 1836 + 1795
            'Male_HighIncome': 1325, # 517 + 808
            'Female_Young': 2236,    # 1980 + 256
            'Female_Old': 2808,      # 2401 + 407
            'Female_LowIncome': 4381,# 1980 + 2401
            'Female_HighIncome': 663,# 256 + 407
            'Young_LowIncome': 3816, # 1836 + 1980
            'Young_HighIncome': 773, # 517 + 256
            'Old_LowIncome': 4196,   # 1795 + 2401
            'Old_HighIncome': 1215,  # 808 + 407
            
            # 1-feature segments 
            'Male': 4956,            # 1836 + 517 + 1795 + 808
            'Female': 5044,          # 1980 + 256 + 2401 + 407
            'Young': 4589,           # 1836 + 517 + 1980 + 256
            'Old': 5411,             # 1795 + 808 + 2401 + 407
            'LowIncome': 8012,       # 1836 + 1795 + 1980 + 2401
            'HighIncome': 1988,      # 517 + 808 + 256 + 407
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
            
            if impressions_won >= campaign.reach:
                continue
            
            remaining_reach = campaign.reach - impressions_won
            
            current_rho = self.effective_reach(impressions_won, campaign.reach)
            target_rho = self.effective_reach(campaign.reach, campaign.reach) 
            delta_rho = target_rho - current_rho
            avg_value_per_impression = (delta_rho * campaign.budget) / remaining_reach if remaining_reach > 0 else 0
            

            marginal_rho = self._marginal_effective_reach(impressions_won, campaign.reach, self.segment_sizes[campaign.target_segment.name])

            #print("self.segment_sizes[campaign.target_segment.name]", self.segment_sizes[campaign.target_segment.name])
            marginal_value_per_impression = marginal_rho * campaign.budget

            optimal_shade = 0.37

            if avg_value_per_impression > 0:
                progress_shade = marginal_value_per_impression / avg_value_per_impression
            else:
                progress_shade = -optimal_shade
            
            
            bid_per_item = avg_value_per_impression * (progress_shade + optimal_shade)

            bid_limit = remaining_budget
            
            # see if its a valid bid
            bid_per_item = min(bid_per_item, bid_limit)
            
            # Ensure bid_per_item is strictly positive for valid bid
            if bid_per_item <= 0 or bid_limit <= 0:
                continue

            #print("bid_per_item", bid_per_item)
            
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
                "marginal_value": float(marginal_value_per_impression),
                "progress_shade": float(progress_shade),
                "market_segment": campaign.target_segment.name,
            })
        
        return bundles

    def get_campaign_bids(self, campaigns_for_auction: Set[Campaign]) -> Dict[Campaign, float]:
        bids = {}

        for campaign in campaigns_for_auction:
            if campaign.end_day - campaign.start_day < 2:
                continue

            active_campaigns = self.get_active_campaigns()
            for active_campaign in active_campaigns:
                if active_campaign.target_segment.name == campaign.target_segment.name:
                    continue

                estimated_bid = self.quality_score * campaign.reach / 0.45



                if self.is_valid_campaign_bid(campaign, estimated_bid):
                    bids[campaign] = estimated_bid
            

            #print("campaign.start_day", campaign.start_day)
            #print("campaign.end_day", campaign.end_day)



        
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
    
    def _marginal_effective_reach(self, x: int, R: int, reach_target: int) -> float:
        """
        Calculate derivative of effective reach function dρ/dx at position (x + reach_target).
        
        This evaluates the marginal value if we were to win reach_target more impressions.
        
        From spec: ρ(C) = (2/a) × [arctan(a×(x/R) - b) - arctan(-b)]
        
        Derivative: dρ/dx = (2/a) × 1/(1 + (a×(x/R) - b)²) × (a/R)
                          = 2/(R × (1 + (a×(x/R) - b)²))
        
        We evaluate at position (x + reach_target) to estimate value at that future point.
        
        The sigmoidal shape means:
        - Early (x << R): low marginal value (flat part of curve)
        - Middle (x ≈ 0.5R): HIGH marginal value (steep part of curve)
        - Near target (x ≈ R): HIGHEST marginal value (approaching ρ=1.0)
        - Beyond (x > R): decreasing marginal value (approaching asymptote at 1.38)
        """
        if R <= 0:
            return 0.0
        
        # Evaluate derivative at position (x + reach_target)
        evaluation_point = x + reach_target
        ratio = evaluation_point / float(R)
        term = self.a * ratio - self.b
        

        denominator = R * (1 + term * term)
        
        if denominator <= 0:
            return 0.0
        
        return 2.0 / denominator
    
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
                    marginal_value = entry.get("marginal_value", 0)
                    progress_shade = entry.get("progress_shade", 1.0)

                    print(
                        f"  Day {day}: bid=${bid_per_item:.3f}, "
                        f"limit=${bid_limit:.2f}, "
                        f"need={remaining_reach} imps, "
                        f"avg=${avg_value:.3f}, marginal=${marginal_value:.3f}, shade={progress_shade:.2f}x"
                    )

        print("=" * 100 + "\n")

if __name__ == "__main__":
    # Here's an opportunity to test offline against some TA agents. Just run this file to do so.
    test_agents = [MyNDaysNCampaignsAgent(),] + [Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)]

    # Don't change this. Adapt initialization to your environment
    simulator = AdXGameSimulator()
    simulator.run_simulation(agents=test_agents, num_simulations=10)