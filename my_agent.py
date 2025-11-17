from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.local_games.adx_arena import AdXGameSimulator
from agt_server.agents.utils.adx.structures import Bid, Campaign, BidBundle, MarketSegment 
from typing import Set, Dict

class MyNDaysNCampaignsAgent(NDaysNCampaignsAgent):

    def __init__(self):
        super().__init__()
        self.name = ""
        
        self.campaign_bid_aggression = 0.4  
        self.spending_buffer = 0.8  

    def on_new_game(self) -> None:
        """Initialize/reset per-game data structures."""
        pass

    def get_ad_bids(self) -> Set[BidBundle]:
        # create ad bids for all active campaigns
        bundles = set()
        
        for campaign in self.get_active_campaigns():
            try:
                # current progress of campaign
                impressions_won = self.get_cumulative_reach(campaign)
                cost_so_far = self.get_cumulative_cost(campaign)
                
                if impressions_won is None:
                    impressions_won = 0
                if cost_so_far is None:
                    cost_so_far = 0
                
                # calculate remaining. max to 1 to avoid division by zero
                remaining_reach = max(1, campaign.reach - impressions_won)
                remaining_budget = max(1.0, campaign.budget * self.spending_buffer - cost_so_far)
                
                # skip if complete or out of budget
                if remaining_reach <= 0 or remaining_budget <= 0:
                    continue
                
                # remaining budget / remaining reach
                # capped between 0.1 and 1.0
                bid_per_item = min(1.0, max(0.1, remaining_budget / remaining_reach))
                
                bid = Bid(
                    bidder=self,
                    auction_item=campaign.target_segment,
                    bid_per_item=bid_per_item,
                    bid_limit=remaining_budget
                )
                
                bid_entries = {bid}
                
                # bundle with just one bid
                bundle = BidBundle(
                    campaign_id=campaign.uid,
                    limit=remaining_budget,
                    bid_entries=bid_entries
                )
                bundles.add(bundle)
                
            except Exception as e:
                # skip if causing errors
                continue
        
        return bundles

    def get_campaign_bids(self, campaigns_for_auction: Set[Campaign]) -> Dict[Campaign, float]:
        # bid on campaigns based on profit potential
        bids = {}
        
        for campaign in campaigns_for_auction:
            try:
                if campaign is None:
                    continue
                
                # estimate cost to fulfill campaign
                base_cpm = 0.15
                
                # adjust cpm based on segment specificity
                # More specific segments = less competition = lower cost
                num_attributes = len(campaign.target_segment.name.split('_'))
                if num_attributes == 1:
                    estimated_cpm = base_cpm * 1.3
                elif num_attributes == 2:
                    estimated_cpm = base_cpm * 1.1  
                else:
                    estimated_cpm = base_cpm  
                
                estimated_impression_cost = campaign.reach * estimated_cpm
                
                # estimate revenue
                # Assume we can fulfill the campaign (reach = impressions won)
                # use effective_reach function to account for sigmoid curve
                rho = self.effective_reach(campaign.reach, campaign.reach)
                if rho is None or rho <= 0:
                    rho = 1.0  
                
                expected_revenue = rho * campaign.budget
                
                # calculate expected profit
                expected_profit = expected_revenue - estimated_impression_cost
                
                # only bid if profitable
                if expected_profit > 0:
                    # bid a fraction of expected profit
                    raw_bid = expected_profit * self.campaign_bid_aggression
                    
                    # aAdjust for quality score
                    # Higher quality score = can bid more and still be competitive
                    # Since effective_bid = bid / quality_score, multiply by quality
                    quality_score = self.get_quality_score()
                    if quality_score is None or quality_score <= 0:
                        quality_score = 1.0
                    
                    # quality adjusted bid
                    bid_value = raw_bid * quality_score
                    
                    # bid at least 10% of reach to have a chance
                    min_bid = campaign.reach * 0.1
                    bid_value = max(bid_value, min_bid)
                    
                    # ensure bid is valid
                    if self.is_valid_campaign_bid(campaign, bid_value):
                        bids[campaign] = bid_value
                    
            except Exception as e:
                # skip if causing errors
                continue
        
        return bids


    def _get_matching_segments(self, target: MarketSegment) -> Set[MarketSegment]:
        # unused rn 
        matching = set()
        
        for segment in MarketSegment.all_segments():
            if segment.issubset(target):
                matching.add(segment)
        
        return matching

    def _estimate_segment_size(self, segment: MarketSegment) -> float:
        # unused rn
        segment_sizes = {
            'Male_Young_LowIncome': 1836,
            'Male_Young_HighIncome': 517,
            'Male_Old_LowIncome': 1795,
            'Male_Old_HighIncome': 808,
            'Female_Young_LowIncome': 1980,
            'Female_Young_HighIncome': 256,
            'Female_Old_LowIncome': 2401,
            'Female_Old_HighIncome': 407,
        }
        
        # for exact matches
        if segment.name in segment_sizes:
            return segment_sizes[segment.name]
        
        # for broader segments, sum up matching segments
        total = 0
        for all_seg in MarketSegment.all_segments():
            if all_seg.name in segment_sizes and all_seg.issubset(segment):
                total += segment_sizes[all_seg.name]
        
        return max(total, 1)  # avoid division by zero

if __name__ == "__main__":
    # Here's an opportunity to test offline against some TA agents. Just run this file to do so.
    test_agents = [MyNDaysNCampaignsAgent()] + [Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)]

    # Don't change this. Adapt initialization to your environment
    simulator = AdXGameSimulator()
    simulator.run_simulation(agents=test_agents, num_simulations=500)
