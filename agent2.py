import math
from agent4 import MyNDaysNCampaignsAgent
from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.local_games.adx_arena import AdXGameSimulator
from agt_server.agents.utils.adx.structures import Bid, Campaign, BidBundle, MarketSegment 
from typing import Set, Dict, List, Tuple



class TrialNDaysNCampaignsAgent(NDaysNCampaignsAgent):

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.max_active_campaigns = 5  # cap here so that we can actually complete campaigns

        # per-campaign final stats
        self.campaign_history: Dict[int, Dict] = {}
        # (day, quality_score)
        self.daily_quality_scores: List[Tuple[int, float]] = []
        # ad-bid history per campaign
        # cid -> list of dict(day, bid_per_item, bid_limit, remaining_reach, remaining_budget, ...)
        self.ad_bid_history: Dict[int, List[Dict]] = {}

        self.active_market_segments = set()

        self.segment_probability = {
            MarketSegment(("Male", "Young")): 0.2353,
            MarketSegment(("Male", "Old")): 0.2603,
            MarketSegment(("Male", "LowIncome")): 0.3631,
            MarketSegment(("Male", "HighIncome")): 0.1325,

            MarketSegment(("Female", "Young")): 0.2236,
            MarketSegment(("Female", "Old")): 0.2808,
            MarketSegment(("Female", "LowIncome")): 0.4381,
            MarketSegment(("Female", "HighIncome")): 0.0663,

            MarketSegment(("Young", "LowIncome")): 0.3816,
            MarketSegment(("Young", "HighIncome")): 0.0773,
            MarketSegment(("Old", "LowIncome")): 0.4196,
            MarketSegment(("Old", "HighIncome")): 0.1215,

            MarketSegment(("Male", "Young", "LowIncome")): 0.1836,
            MarketSegment(("Male", "Young", "HighIncome")): 0.0517,
            MarketSegment(("Male", "Old", "LowIncome")): 0.1795,
            MarketSegment(("Male", "Old", "HighIncome")): 0.0808,

            MarketSegment(("Female", "Young", "LowIncome")): 0.1980,
            MarketSegment(("Female", "Young", "HighIncome")): 0.0256,
            MarketSegment(("Female", "Old", "LowIncome")): 0.2401,
            MarketSegment(("Female", "Old", "HighIncome")): 0.0407,
        }


    def on_new_game(self) -> None:
        """Initialize/reset per-game data structures."""
        super().on_new_game()
        self.campaign_history.clear()
        self.daily_quality_scores.clear()
        self.ad_bid_history.clear()

    def generate_ad_bid(self, campaign, current_day) -> float:
        impressions_won = self.get_cumulative_reach(campaign) or 0
        cost_so_far = self.get_cumulative_cost(campaign) or 0
        campaign_segment = campaign.target_segment
        segment_prob = self.segment_probability.get(campaign_segment, 0.0)
        if segment_prob == 0.0:
            print("bad Segment prob:", campaign_segment) 
        expected_impressions_today = 10000 * segment_prob

        immediate_marginal_value_of_impression = (self.effective_reach(impressions_won + 1, campaign.reach) - self.effective_reach(impressions_won, campaign.reach)) * campaign.budget
        half_expected_marginal = (self.effective_reach(impressions_won + int(0.5 * expected_impressions_today), campaign.reach) - self.effective_reach(impressions_won, campaign.reach)) * campaign.budget / (0.5 * expected_impressions_today)

        if impressions_won / campaign.reach < 0.5:
            bid_per_item = 0.9 * half_expected_marginal
        else:
            bid_per_item = 0.85 * immediate_marginal_value_of_impression

        return bid_per_item

    def get_ad_bids(self) -> Set[BidBundle]:
        # create ad bids for all active campaigns
        bundles: Set[BidBundle] = set()

        # update campaign / quality tracking once per day
        self._print_daily_campaign_status()

        current_day = self.get_current_day()
        for campaign in self.get_active_campaigns():
            impressions_won = self.get_cumulative_reach(campaign) or 0
            cost_so_far = self.get_cumulative_cost(campaign) or 0
            remaining_reach = max(0, campaign.reach - impressions_won)
            remaining_budget = max(0.0, campaign.budget - cost_so_far)
            bid_per_item = self.generate_ad_bid(campaign, current_day)

            if remaining_reach == 0 or remaining_budget <= 0 or bid_per_item <= 0:
                continue
            
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
                "market_segment": campaign.target_segment.name,
            })
        
        return bundles

    def get_campaign_bids(self, campaigns_for_auction: Set[Campaign]) -> Dict[Campaign, float]:
        """Bid in the campaign-level auction (to win the right to serve)."""
        bids: Dict[Campaign, float] = {}
        
        current_day = self.get_current_day()

        for campaign in self.get_active_campaigns():
            if self.get_cumulative_reach(campaign) < campaign.reach:
                self.active_market_segments.add(campaign.target_segment)
        
        for campaign in campaigns_for_auction:
            campaign_segment = campaign.target_segment
            # skip segments we are not interested in
            if campaign_segment in self.active_market_segments:
                continue
            campaign_duration = (campaign.end_day - campaign.start_day + 1) # 1-3 days
            expected_number_impressions = 100000 * self.segment_probability.get(campaign_segment, 0.0) * campaign_duration
            reach_ratio = campaign.reach / expected_number_impressions

            if reach_ratio > 0.5:
                f = 1
            elif reach_ratio <= 0.33:
                f =  0.35 
            else:
                f = 0.75
            
            effective_target = f * campaign.reach
            raw_bid = effective_target * self.get_quality_score()
            bid_value = self.clip_campaign_bid(campaign, raw_bid)
            if self.is_valid_campaign_bid(campaign, bid_value):
                bids[campaign] = bid_value

        # if self.name == "big bidder":
        #     print("Day", current_day, "Campaign bids:", {c.uid: b for c, b in bids.items()})
        return bids

    def _print_daily_campaign_status(self):
        """
        Track daily quality score and snapshot campaign stats into campaign_history.
        (No printing/logging – just bookkeeping.)
        """
        current_day = self.get_current_day()
        active_campaigns = list(self.get_active_campaigns())
    
        quality_score = self.get_quality_score()
        self.daily_quality_scores.append((current_day, quality_score))
        
        for campaign in active_campaigns:
            impressions_won = self.get_cumulative_reach(campaign) or 0
            cost_so_far = self.get_cumulative_cost(campaign) or 0
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
        print("\n" + "=" * 100)
        print(f"POST-GAME SUMMARY for {self.name}")
        print("=" * 100)

        print("\nQuality score by day:")
        for day, q in self.daily_quality_scores:
            print(f"  Day {day}: Q = {q:.4f}")

        print("\nCampaign outcomes:")
        for cid, info in sorted(self.campaign_history.items()):
            R = info["reach"]
            B = info["budget"]
            x = info["final_impressions"]
            k = info["final_cost"]
            rho = self.effective_reach(x, R) if R >= 0 else 0.0
            approx_profit = rho * B - k

            print(
                f"  Campaign {cid} [{info['segment']}]: "
                f"reach={R}, budget={B:.2f}, "
                f"impressions={x}, cost={k:.2f}, "
                f"rho={rho:.3f}, approx profit={approx_profit:.2f}, "
                f"days={info['start_day']}–{info['end_day']}"
                )


        for cid, bids in self.ad_bid_history.items():
            print(f"\nAd bid history for Campaign {cid}:")
            for entry in bids:
                day = entry["day"]
                bid_per_item = entry["bid_per_item"]
                bid_limit = entry["bid_limit"]
                remaining_reach = entry["remaining_reach"]
                remaining_budget = entry["remaining_budget"]
                market_segment = entry["market_segment"]

                print(
                    f"  Day {day}: bid_per_item={bid_per_item:.3f}, "
                    f"bid_limit={bid_limit:.2f}, "
                    f"remaining_reach={remaining_reach}, "
                    f"remaining_budget={remaining_budget:.2f}, "
                    f" (segment={market_segment})"
                )

        print("=" * 100 + "\n")


if __name__ == "__main__":
    my_agent = TrialNDaysNCampaignsAgent(name="big bidder")
    # zach_agent = MyNDaysNCampaignsAgent()

    # test_agents = [my_agent, zach_agent] + [Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)]

    zach_agents = [MyNDaysNCampaignsAgent(name=f"Zach Agent {i + 1}") for i in range(5)]
    my_agents = [TrialNDaysNCampaignsAgent(name=f"Derek Agent {i + 1}") for i in range(4)]
    test_agents = [my_agent] + zach_agents + my_agents

    # test_agents = [my_agent] + [TrialNDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)]

    simulator = AdXGameSimulator()
    simulator.run_simulation(agents=test_agents, num_simulations=20)
    my_agent.print_debug_summary()
