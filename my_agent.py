from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.local_games.adx_arena import AdXGameSimulator
from agt_server.agents.utils.adx.structures import Bid, Campaign, BidBundle, MarketSegment 
from typing import Set, Dict, List, Tuple, Optional
import collections
# toggle logging
ENABLE_DEBUG_LOGGING = True 

class MyNDaysNCampaignsAgent(NDaysNCampaignsAgent):

    def __init__(self):
        super().__init__()
        self.name = "big bidder"
        
        self.campaign_bid_aggression = 0.75
        self.spending_buffer = 0.90
        self.max_bid_multiplier = 0.45  # Cap at reach × 0.45 (below Tier1 average)
        
        self.max_active_campaigns = 4 # cap here so that we can actually complete campaigns


        # Logging - use single file for all games
        self.log_filename = "agent_debug.txt"
        self.log_file = None
        self.log_enabled = ENABLE_DEBUG_LOGGING  # Use the global toggle  

        self.campaign_history: Dict[int, Dict] = {}
        self.daily_quality_scores: List[Tuple[int, float]] = []

        self.ad_bid_history: Dict[int, List[Dict]] = {}



    def on_new_game(self) -> None:
        """Initialize/reset per-game data structures."""
        self.last_debug_day = -1

        self.campaign_history.clear()
        self.daily_quality_scores.clear()
        self.ad_bid_history.clear()  
        
        # Open/append to single log file
        if self.log_enabled and self.log_file is None:
            self.log_file = open(self.log_filename, 'a')
        

        self._log(f"\n\n{'='*80}")
        self._log(f"NEW GAME STARTED - Agent: {self.name}")
        self._log(f"{'='*80}\n")

    def get_ad_bids(self) -> Set[BidBundle]:
        # create ad bids for all active campaigns
        bundles = set()
        
        # Print daily campaign status
        current_day = self.get_current_day()
        if current_day != self.last_debug_day:
            self._print_daily_campaign_status()
            self.last_debug_day = current_day
        
        self._log(f"\n--- AD BIDDING (Day {current_day}) ---")
        
        for campaign in self.get_active_campaigns():
            try:
                # current progress of campaign
                impressions_won = self.get_cumulative_reach(campaign)
                cost_so_far = self.get_cumulative_cost(campaign)
                
                if impressions_won is None:
                    impressions_won = 0
                if cost_so_far is None:
                    cost_so_far = 0
                
                # calculate remaining
                remaining_reach = campaign.reach - impressions_won
                remaining_budget = campaign.budget * 0.9 - cost_so_far
                
                # stop if complete or out of budget
                if remaining_reach <= 0:
                    self._log(f"  [SKIP] Campaign {campaign.uid}: segment={campaign.target_segment.name} "
                          f"(complete)")
                    continue
                if remaining_budget <= 0:
                    self._log(f"  [SKIP] Campaign {campaign.uid}: segment={campaign.target_segment.name} "
                          f"(no budget)")
                    continue
                
                # bid VERY aggressively to actually win impressions
                budget_ratio = campaign.budget / campaign.reach
                
                # calculate how many impressions we need per day to complete
                current_day = self.get_current_day()
                days_left = max(1, campaign.end_day - current_day)
                impressions_needed_per_day = remaining_reach / days_left
                
                # if we are behind schedule, bid MORE aggressively
                completion_pct = (impressions_won / campaign.reach) if campaign.reach > 0 else 0
                days_elapsed = current_day - campaign.start_day
                expected_completion = days_elapsed / (campaign.end_day - campaign.start_day) if (campaign.end_day - campaign.start_day) > 0 else 0
                
                if completion_pct < expected_completion:
                    # if we are behind schedule, bid MUCH higher (60% premium)
                    urgency_multiplier = 1.33
                    self._log(f"        [URGENT] Behind schedule: {completion_pct:.0%} vs {expected_completion:.0%} expected")
                elif days_left <= 1:
                    # if it's the last day, be very aggressive (70% premium)
                    urgency_multiplier = 1.5
                    self._log(f"        [FINAL DAY] Last chance to complete!")
                else:
                    # if we are on track, bid 40% higher than budget ratio
                    urgency_multiplier = 1.0

                aggressive_bid = (budget_ratio ** 3) * (urgency_multiplier ** 2)

                # still need to respect remaining budget
                safe_remaining_reach = max(1, remaining_reach)
                max_safe_bid = remaining_budget / safe_remaining_reach

                
                # use aggressive bid, but don't overspend
                bid_per_item = min(aggressive_bid, max_safe_bid)
                
                # cap at reasonable bounds (capped to 3)
                bid_per_item = min(3.0, max(0.15, bid_per_item))
                
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
                
                # Debug output
                self._log(f"  [BID] Campaign {campaign.uid}: segment={campaign.target_segment.name}")
                self._log(f"        Bid/item: ${bid_per_item:.4f}, Limit: ${remaining_budget:.2f}")
                self._log(f"        Progress: {impressions_won}/{campaign.reach} impressions ({completion_pct:.0%}), "
                      f"${cost_so_far:.2f}/${campaign.budget:.2f} spent")
                self._log(f"        Days left: {days_left}, Need: {impressions_needed_per_day:.0f} impr/day")
                
            except Exception as e:
                # skip if causing errors
                self._log(f"  [ERROR] Campaign processing error: {e}")
                continue
        
        self._log(f"Total ad bid bundles created: {len(bundles)}\n")
        return bundles

    def get_campaign_bids(self, campaigns_for_auction: Set[Campaign]) -> Dict[Campaign, float]:
        # bid on campaigns based on profit potential
        bids = {}
        
        current_day = self.get_current_day()
        active_count = len(list(self.get_active_campaigns()))
        
        self._log(f"\n--- CAMPAIGN AUCTION (Day {current_day}) ---")
        self._log(f"Campaigns available for bidding: {len(campaigns_for_auction)}")
        self._log(f"Currently active campaigns: {active_count}/{self.max_active_campaigns}")
        
        # don't bid on new campaigns if we're at capacity
        if active_count >= self.max_active_campaigns:
            self._log(f"[SKIP ALL] Already at max capacity ({self.max_active_campaigns} active campaigns)")
            self._log(f"{'='*60}\n")
            return bids
        
        # limit how many new campaigns we can take on
        max_new_campaigns = self.max_active_campaigns - active_count
        campaigns_bid_count = 0
        
        for campaign in campaigns_for_auction:
            # stop bidding if we've reached our limit for this auction
            if campaigns_bid_count >= max_new_campaigns:
                self._log(f"\n[LIMIT REACHED] Already bidding on {campaigns_bid_count} campaigns this auction")
                break
                
            try:
                if campaign is None:
                    continue
                
                # check campaign duration - prioritize longer campaigns
                campaign_duration = campaign.end_day - campaign.start_day
                if campaign_duration < 2:
                    self._log(f"\n  [SKIP] Campaign: {campaign.target_segment.name}")
                    self._log(f"    Duration: {campaign_duration} days (need 2+ days)")
                    continue
                
                # check if campaign is realistically completable
                # for 2-day campaigns, we need to be more aggressive
                # assume we can win ~30% of impressions per day with aggressive bidding
                impressions_per_day = campaign.reach * 0.30
                total_possible = impressions_per_day * campaign_duration
                completion_ratio = total_possible / campaign.reach
                
                # for 2-day campaigns, lower the bar to 60% since time is tight
                min_completion = 0.60
                if completion_ratio < min_completion:
                    self._log(f"\n  [SKIP] Campaign: {campaign.target_segment.name}")
                    self._log(f"    Reach: {campaign.reach}, Duration: {campaign_duration} days")
                    self._log(f"    Estimated completion: {completion_ratio*100:.0f}% (need {min_completion*100:.0f}%+)")
                    continue
                
                base_cpm = 0.025  # Lower CPM estimate
                
                # adjust CPM based on segment specificity
                num_attributes = len(campaign.target_segment.name.split('_'))
                if num_attributes == 1:
                    segment_multiplier = 1.4
                elif num_attributes == 2:
                    segment_multiplier = 1.15
                else:
                    segment_multiplier = 1.0
                
                estimated_cpm = base_cpm * segment_multiplier
                estimated_cost = campaign.reach * estimated_cpm
                
                # assume perfect fulfillment
                rho = 1.0
                
                # calculate bid: cost + profit margin
                profit_margin = 1.5  # 50% profit target
                base_bid_value = (estimated_cost / rho) * profit_margin
                
                # apply aggression
                bid_value = base_bid_value * self.campaign_bid_aggression
                
                # apply quality score
                quality_score = self.get_quality_score()
                if quality_score is None or quality_score <= 0:
                    quality_score = 1.0
                bid_value = bid_value * quality_score
                
                # keep within valid range [0.1 × reach, 1.0 × reach]
                # cap at max_bid_multiplier to win (lower bids win!)
                min_valid_bid = campaign.reach * 0.1  # API requirement
                max_competitive_bid = campaign.reach * self.max_bid_multiplier
                bid_value = max(min_valid_bid, min(bid_value, max_competitive_bid))
                
                self._log(f"\n  Campaign Analysis:")
                self._log(f"    Segment: {campaign.target_segment.name}")
                self._log(f"    Reach: {campaign.reach}, Duration: {campaign_duration} days")
                self._log(f"    Est. Cost: ${estimated_cost:.2f}, Completability: {completion_ratio:.0%}")
                self._log(f"    Bid range: [${min_valid_bid:.2f}, ${max_competitive_bid:.2f}]")
                self._log(f"    Final bid: ${bid_value:.2f} (Quality: {quality_score:.2f})")
                
                # ensure bid is valid and place it
                if self.is_valid_campaign_bid(campaign, bid_value):
                    bids[campaign] = bid_value
                    campaigns_bid_count += 1
                    self._log(f"    [BIDDING]")
                else:
                    self._log(f"    [INVALID BID]")
                    
            except Exception as e:
                # skip if causing errors
                self._log(f"  [ERROR] Campaign evaluation error: {e}")
                continue
        
        self._log(f"\n{'='*60}")
        self._log(f"SUMMARY: Bidding on {len(bids)}/{len(campaigns_for_auction)} campaigns")
        self._log(f"{'='*60}\n")
        return bids


    def _log(self, message):
        """Print to console and write to log file (only if logging enabled)."""
        if not self.log_enabled:
            return  # skip all logging if disabled
        
        print(message)
        if self.log_file:
            self.log_file.write(str(message) + '\n')
            self.log_file.flush()  # Ensure it's written immediately

    def _print_daily_campaign_status(self):
        """Print the status of all active campaigns at the start of each day."""
        current_day = self.get_current_day()
        active_campaigns = list(self.get_active_campaigns())
        
        # Get quality score
        quality_score = self.get_quality_score()
        if quality_score is None:
            quality_score = 1.0

        if not self.daily_quality_scores or self.daily_quality_scores[-1][0] != current_day:
            self.daily_quality_scores.append((current_day, quality_score))
        
        self._log(f"\n{'='*80}")
        self._log(f"DAILY CAMPAIGN STATUS - Day {current_day}")
        self._log(f"{'='*80}")
        self._log(f"Quality Score: {quality_score:.4f}")
        self._log(f"Active Campaigns: {len(active_campaigns)}")
        
        if not active_campaigns:
            self._log("  No active campaigns")
        else:
            total_budget = 0
            total_spent = 0
            total_reach_target = 0
            total_impressions = 0
            
            for campaign in active_campaigns:
                try:
                    impressions_won = self.get_cumulative_reach(campaign) or 0
                    cost_so_far = self.get_cumulative_cost(campaign) or 0
                    
                    total_budget += campaign.budget
                    total_spent += cost_so_far
                    total_reach_target += campaign.reach
                    total_impressions += impressions_won
                    
                    completion_pct = (impressions_won / campaign.reach * 100) if campaign.reach > 0 else 0
                    budget_usage_pct = (cost_so_far / campaign.budget * 100) if campaign.budget > 0 else 0

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
                    
                    self._log(f"\n  Campaign {campaign.uid}:")
                    self._log(f"    Target Segment: {campaign.target_segment.name}")
                    self._log(f"    Progress: {impressions_won:,}/{campaign.reach:,} impressions ({completion_pct:.1f}%)")
                    self._log(f"    Budget: ${cost_so_far:.2f}/${campaign.budget:.2f} ({budget_usage_pct:.1f}%)")
                    self._log(f"    Days Remaining: {campaign.end_day - current_day}")
                    
                    # Calculate effective CPM
                    if impressions_won > 0:
                        cpm = (cost_so_far / impressions_won) * 1000
                        self._log(f"    Effective CPM: ${cpm:.4f}")
                    
                except Exception as e:
                    self._log(f"  Campaign {campaign.uid}: Error reading status - {e}")
            
            self._log(f"\n  {'─'*76}")
            self._log(f"  TOTALS:")
            self._log(f"    Budget: ${total_spent:.2f}/${total_budget:.2f}")
            self._log(f"    Impressions: {total_impressions:,}/{total_reach_target:,}")
            if total_impressions > 0:
                avg_cpm = (total_spent / total_impressions) * 1000
                self._log(f"    Average CPM: ${avg_cpm:.4f}")
        
        self._log(f"{'='*80}\n")

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

    def print_debug_summary(self):
        """Print a post-game summary of campaigns, spend, fill, and Q over time."""
        self._log("\n" + "="*80)
        self._log(f"POST-GAME SUMMARY for {self.name}")
        self._log("="*80)

        if self.daily_quality_scores:
            self._log("\nQuality score by day:")
            for day, q in self.daily_quality_scores:
                self._log(f"  Day {day}: Q = {q:.4f}")
        else:
            self._log("\nNo quality score history recorded.")

        # Campaign outcomes
        if self.campaign_history:
            self._log("\nCampaign outcomes:")
            for cid, info in sorted(self.campaign_history.items()):
                R = info["reach"]
                B = info["budget"]
                x = info["final_impressions"]
                k = info["final_cost"]
                rho = self.effective_reach(x, R) if R > 0 else 0.0
                approx_profit = rho * B - k

                self._log(
                    f"  Campaign {cid} [{info['segment']}]: "
                    f"reach={R}, budget={B:.2f}, "
                    f"impressions={x}, cost={k:.2f}, "
                    f"rho={rho:.3f}, approx profit={approx_profit:.2f}, "
                    f"days={info['start_day']}–{info['end_day']}"
                )
                # if cid in self.ad_bid_history:
                #     bids_for_c = self.ad_bid_history[cid]
                #     num_days_bid = len(bids_for_c)
                #     avg_bid = sum(b["bid_per_item"] for b in bids_for_c) / num_days_bid
                #     max_bid = max(b["bid_per_item"] for b in bids_for_c)
                #     min_bid = min(b["bid_per_item"] for b in bids_for_c)

                #     self._log(
                #         f"    Ad-bids: {num_days_bid} entries, "
                #         f"avg={avg_bid:.3f}, min={min_bid:.3f}, max={max_bid:.3f}"
                #     )
                #     for b in bids_for_c:
                #         self._log(
                #             f"      Day {b['day']}: bid_per_item={b['bid_per_item']:.3f}, "
                #             f"rem_reach={b['remaining_reach']}, "
                #             f"rem_budget={b['remaining_budget']:.2f}"
                #         )

        else:
            self._log("\nNo campaign history recorded.")

        self._log("="*80 + "\n")

if __name__ == "__main__":
    # Here's an opportunity to test offline against some TA agents. Just run this file to do so.
    my_agent = MyNDaysNCampaignsAgent()
    test_agents = [my_agent] + [Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)]

    # Don't change this. Adapt initialization to your environment
    simulator = AdXGameSimulator()
    simulator.run_simulation(agents=test_agents, num_simulations=1)
    my_agent.print_debug_summary()

