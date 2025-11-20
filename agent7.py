from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.local_games.adx_arena import AdXGameSimulator
from agt_server.agents.utils.adx.structures import Bid, Campaign, BidBundle, MarketSegment 
from typing import Set, Dict
import itertools
import numpy as np

class BasicBot(NDaysNCampaignsAgent):

    def __init__(self, name = "Basic Bot"):
        # TODO: fill this in (if necessary)
        super().__init__()
        self.name = name  

    def on_new_game(self) -> None:
        # TODO: fill this in (if necessary)
        pass

    def get_ad_bids(self) -> Set[BidBundle]:
        # TODO: fill this in
        bundles = set()

        for campaign in self.get_active_campaigns():
            shade = campaign_utils.campaign_shade(campaign, self.get_active_campaigns())
            budget_remaining = campaign.budget - campaign.cumulative_cost
            reach_remaining = campaign.reach - campaign.cumulative_reach
            bid_per_item = shade * (budget_remaining) / (reach_remaining) if reach_remaining != 0 else 0
            
            bid_entries = set()
            bid_entries.add(Bid(bidder=self, auction_item=campaign.target_segment, bid_per_item=bid_per_item, bid_limit=budget_remaining))
            
            bundle = BidBundle(campaign_id=campaign.uid, limit=budget_remaining, bid_entries=bid_entries)
            bundles.add(bundle)
        return bundles
    
    def get_campaign_bids(self, campaigns_for_auction:  Set[Campaign]) -> Dict[Campaign, float]:
        bids = {}
        for campaign in campaigns_for_auction:
            market_segment = campaign.target_segment
            market_segment_population = CONFIG['market_segment_pop'][market_segment]
            percentage_of_population = campaign.reach / market_segment_population
            if abs(percentage_of_population - 0.3) < 0.01:
                shading = 0.8 / self.quality_score
            elif abs(percentage_of_population - 0.5) < 0.01:
                shading = 0.9 / self.quality_score
            elif abs(percentage_of_population - 0.7) < 0.01:
                shading = 1 / self.quality_score
            else:
                # ERROR SHOULD NOT BE HERE
                print("SHOULD NOT REACH")
                shading = 1 / self.quality_score
            bid = campaign.reach * shading
            bids[campaign] = bid
        return bids

if __name__ == "__main__":
    # Here's an opportunity to test offline against some TA agents. Just run this file to do so.
    test_agents = [MyNDaysNCampaignsAgent()] + [Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)]

    # Don't change this. Adapt initialization to your environment
    simulator = AdXGameSimulator()
    simulator.run_simulation(agents=test_agents, num_simulations=100)