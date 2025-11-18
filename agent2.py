from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.local_games.adx_arena import AdXGameSimulator
from agt_server.agents.utils.adx.structures import Bid, Campaign, BidBundle, MarketSegment 
from typing import Set, Dict, List, Tuple


class TrialNDaysNCampaignsAgent(NDaysNCampaignsAgent):

    def __init__(self):
        super().__init__()
        self.name = "big bidder"
        
        self.campaign_bid_aggression = 0.75
        self.max_bid_multiplier = 0.45  # Cap at reach × 0.45 (below Tier1 average)
        self.max_active_campaigns = 4   # cap here so that we can actually complete campaigns

        # per-campaign final stats
        self.campaign_history: Dict[int, Dict] = {}
        # (day, quality_score)
        self.daily_quality_scores: List[Tuple[int, float]] = []
        # ad-bid history per campaign
        # cid -> list of dict(day, bid_per_item, bid_limit, remaining_reach, remaining_budget, ...)
        self.ad_bid_history: Dict[int, List[Dict]] = {}


        self.segment_probability: Dict[MarketSegment, Dict] = {}

    def on_new_game(self) -> None:
        """Initialize/reset per-game data structures."""
        super().on_new_game()
        self.campaign_history.clear()
        self.daily_quality_scores.clear()
        self.ad_bid_history.clear()

    def get_ad_bids(self) -> Set[BidBundle]:
        # create ad bids for all active campaigns
        bundles: Set[BidBundle] = set()

        # update campaign / quality tracking once per day
        self._print_daily_campaign_status()

        current_day = self.get_current_day()
        for campaign in self.get_active_campaigns():
            impressions_won = self.get_cumulative_reach(campaign) or 0
            cost_so_far = self.get_cumulative_cost(campaign) or 0
            
            remaining_reach = campaign.reach - impressions_won
            remaining_budget = campaign.budget * 0.9 - cost_so_far
            
            if remaining_budget <= 0 or remaining_reach <= 0:
                continue
            
            # base "value" per impression = budget/reach
            budget_ratio = campaign.budget / campaign.reach
            
            # schedule info
            days_left = max(1, campaign.end_day - current_day)
            completion_pct = impressions_won / campaign.reach if campaign.reach > 0 else 0.0
            days_elapsed = current_day - campaign.start_day
            total_days = max(1, campaign.end_day - campaign.start_day)
            expected_completion = days_elapsed / total_days

            # urgency heuristic
            if completion_pct < expected_completion:
                urgency_multiplier = 1.33
            elif days_left <= 1:
                urgency_multiplier = 1.5
            else:
                urgency_multiplier = 1.0

            # “aggressive” target bid scaled by budget_ratio
            aggressive_bid = (budget_ratio ** 2) * (urgency_multiplier ** 2)

            # but we still must respect remaining budget
            safe_remaining_reach = max(1, remaining_reach)
            max_safe_bid = remaining_budget / safe_remaining_reach

            bid_per_item = min(aggressive_bid, max_safe_bid)
            bid_per_item = min(3.0, max(max_safe_bid / 2, bid_per_item))  # clamp to [0.15, 3.0]
            bid_per_item = max_safe_bid
            
            
            bid = Bid(
                bidder=self,
                auction_item=campaign.target_segment,
                bid_per_item=bid_per_item,
                bid_limit=remaining_budget
            )
            
            bid_entries = {bid}
            bundle = BidBundle(
                campaign_id=campaign.uid,
                limit=remaining_budget,
                bid_entries=bid_entries
            )
            bundles.add(bundle)

            # record ad-bid for this campaign
            cid = campaign.uid
            if cid not in self.ad_bid_history:
                self.ad_bid_history[cid] = []
            self.ad_bid_history[cid].append({
                "day": current_day,
                "bid_per_item": float(bid_per_item),
                "bid_limit": float(remaining_budget),
                "remaining_reach": int(remaining_reach),
                "remaining_budget": float(remaining_budget),
                "aggressive_bid": float(aggressive_bid),
                "max_safe_bid": float(max_safe_bid),
            })
        
        return bundles

    def get_campaign_bids(self, campaigns_for_auction: Set[Campaign]) -> Dict[Campaign, float]:
        """Bid in the campaign-level auction (to win the right to serve)."""
        bids: Dict[Campaign, float] = {}
        
        current_day = self.get_current_day()
        active_count = len(list(self.get_active_campaigns()))

        # don't bid on new campaigns if we're at capacity
        if active_count >= self.max_active_campaigns:
            return bids
        
        max_new_campaigns = self.max_active_campaigns - active_count
        campaigns_bid_count = 0
        
        for campaign in campaigns_for_auction:
            if campaigns_bid_count >= max_new_campaigns:
                break
            if campaign is None:
                continue
        
            campaign_duration = campaign.end_day - campaign.start_day
            if campaign_duration < 2:
                continue
            
            # estimated completability
            impressions_per_day = campaign.reach * 0.30
            total_possible = impressions_per_day * campaign_duration
            completion_ratio = total_possible / campaign.reach
            
            if completion_ratio < 0.60:
                continue
            
            base_cpm = 0.025
            
            num_attributes = len(campaign.target_segment.name.split('_'))
            if num_attributes == 1:
                segment_multiplier = 1.4
            elif num_attributes == 2:
                segment_multiplier = 1.15
            else:
                segment_multiplier = 1.0
            
            estimated_cpm = base_cpm * segment_multiplier
            estimated_cost = campaign.reach * estimated_cpm
            
            rho = 1.0
            profit_margin = 1.5
            base_bid_value = (estimated_cost / rho) * profit_margin
            
            bid_value = base_bid_value * self.campaign_bid_aggression
            
            quality_score = self.get_quality_score() or 1.0
            bid_value *= quality_score
            
            min_valid_bid = campaign.reach * 0.1
            max_competitive_bid = campaign.reach * self.max_bid_multiplier
            bid_value = max(min_valid_bid, min(bid_value, max_competitive_bid))
            
            if self.is_valid_campaign_bid(campaign, bid_value):
                bids[campaign] = bid_value
                campaigns_bid_count += 1
                
        return bids

    def _print_daily_campaign_status(self):
        """
        Track daily quality score and snapshot campaign stats into campaign_history.
        (No printing/logging – just bookkeeping.)
        """
        current_day = self.get_current_day()
        active_campaigns = list(self.get_active_campaigns())
        
        quality_score = self.get_quality_score() or 1.0

        # record one quality score per day
        if not self.daily_quality_scores or self.daily_quality_scores[-1][0] != current_day:
            self.daily_quality_scores.append((current_day, quality_score))
        
        # always update campaign_history snapshot
        total_budget = 0.0
        total_spent = 0.0
        total_reach_target = 0
        total_impressions = 0
        
        for campaign in active_campaigns:
            impressions_won = self.get_cumulative_reach(campaign) or 0
            cost_so_far = self.get_cumulative_cost(campaign) or 0
            
            total_budget += campaign.budget
            total_spent += cost_so_far
            total_reach_target += campaign.reach
            total_impressions += impressions_won

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

    def print_debug_summary(self):
        print("\n" + "=" * 80)
        print(f"POST-GAME SUMMARY for {self.name}")
        print("=" * 80)

        # quality scores
        if self.daily_quality_scores:
            print("\nQuality score by day:")
            for day, q in self.daily_quality_scores:
                print(f"  Day {day}: Q = {q:.4f}")
        else:
            print("\nNo quality score history recorded.")

        # campaign outcomes + bid stats
        if self.campaign_history:
            print("\nCampaign outcomes:")
            for cid, info in sorted(self.campaign_history.items()):
                R = info["reach"]
                B = info["budget"]
                x = info["final_impressions"]
                k = info["final_cost"]
                rho = self.effective_reach(x, R) if R > 0 else 0.0
                approx_profit = rho * B - k

                print(
                    f"  Campaign {cid} [{info['segment']}]: "
                    f"reach={R}, budget={B:.2f}, "
                    f"impressions={x}, cost={k:.2f}, "
                    f"rho={rho:.3f}, approx profit={approx_profit:.2f}, "
                    f"days={info['start_day']}–{info['end_day']}"
                )
        else:
            print("\nNo campaign history recorded.")

        for cid, bids in self.ad_bid_history.items():
            print(f"\nAd bid history for Campaign {cid}:")
            for entry in bids:
                day = entry["day"]
                bid_per_item = entry["bid_per_item"]
                bid_limit = entry["bid_limit"]
                remaining_reach = entry["remaining_reach"]
                remaining_budget = entry["remaining_budget"]
                aggressive_bid = entry["aggressive_bid"]
                max_safe_bid = entry["max_safe_bid"]

                print(
                    f"  Day {day}: bid_per_item={bid_per_item:.3f}, "
                    f"bid_limit={bid_limit:.2f}, "
                    f"remaining_reach={remaining_reach}, "
                    f"remaining_budget={remaining_budget:.2f}, "
                    f"aggressive_bid={aggressive_bid:.3f}, "
                    f"max_safe_bid={max_safe_bid:.3f}"
                )

        print("=" * 80 + "\n")


if __name__ == "__main__":
    my_agent = TrialNDaysNCampaignsAgent()
    test_agents = [my_agent] + [Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)]

    simulator = AdXGameSimulator()
    simulator.run_simulation(agents=test_agents, num_simulations=1)
    my_agent.print_debug_summary()
