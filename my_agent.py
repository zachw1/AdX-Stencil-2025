from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.local_games.adx_arena import AdXGameSimulator
from agt_server.agents.utils.adx.structures import Bid, Campaign, BidBundle, MarketSegment
from typing import Set, Dict, List, Tuple, Optional
import collections
import math
from agent4 import Agent4

# Toggle logging
ENABLE_DEBUG_LOGGING = True


class MyNDaysNCampaignsAgent(NDaysNCampaignsAgent):

    def __init__(self):
        super().__init__()
        self.name = "smart bidder"

        # Campaign management
        self.max_active_campaigns = 4  # Limit concurrent campaigns to ensure completion

        # These are no longer used directly as hard-coded multipliers, but we keep them
        # in case you want to tweak global aggressiveness later.
        self.urgency_on_track = 1.0
        self.urgency_behind = 1.5
        self.urgency_final_day = 2.0

        # Logging - use single file for all games
        self.log_filename = "agent_debug.txt"
        self.log_file = None
        self.log_enabled = ENABLE_DEBUG_LOGGING  # Use the global toggle

        self.campaign_history: Dict[int, Dict] = {}
        self.daily_quality_scores: List[Tuple[int, float]] = []

        self.ad_bid_history: Dict[int, List[Dict]] = {}

    # -------------------------------------------------------------------------
    # Game lifecycle
    # -------------------------------------------------------------------------

    def on_new_game(self) -> None:
        """Initialize/reset per-game data structures."""
        self.last_debug_day = -1

        self.campaign_history.clear()
        self.daily_quality_scores.clear()
        self.ad_bid_history.clear()

        # Open/append to single log file
        if self.log_enabled and self.log_file is None:
            self.log_file = open(self.log_filename, 'a')

        self._log(f"\n\n{'=' * 80}")
        self._log(f"NEW GAME STARTED - Agent: {self.name}")
        self._log(f"{'=' * 80}\n")

    # -------------------------------------------------------------------------
    # AD BIDDING: per-impression auctions
    # -------------------------------------------------------------------------

    def get_ad_bids(self) -> Set[BidBundle]:
        """
        For each active campaign, bid approximately the *marginal value* of an impression
        adjusted by urgency, rather than a fixed budget/reach ratio.

        Core ideas:
        - Aim for ~95% of target reach (diminishing returns after that).
        - Base bid = (remaining budget) / (remaining useful impressions).
        - Urgency factor depending on how far behind schedule we are.
        - Smooth daily spending with a per-day budget limit.
        """
        bundles: Set[BidBundle] = set()

        current_day = self.get_current_day()
        if current_day != self.last_debug_day:
            self._print_daily_campaign_status()
            self.last_debug_day = current_day

        self._log(f"\n--- AD BIDDING (Day {current_day}) ---")

        for campaign in self.get_active_campaigns():
            try:
                impressions_won = self.get_cumulative_reach(campaign) or 0
                cost_so_far = self.get_cumulative_cost(campaign) or 0.0

                # If we've already spent (or over-spent) the budget, or massively over-fulfilled, stop
                total_budget = campaign.budget
                if total_budget is None:
                    total_budget = 0.0

                remaining_budget = max(0.0, total_budget - cost_so_far)
                if remaining_budget <= 0.0:
                    self._log(f"  [SKIP] Campaign {campaign.uid}: {campaign.target_segment.name} (no budget left)")
                    continue

                # Target reach < full reach because rho() flattens after ~R
                target_reach = int(round(0.95 * campaign.reach))
                target_reach = max(1, target_reach)

                # If we've hit target reach already, don't chase extra expensive impressions
                if impressions_won >= target_reach:
                    self._log(
                        f"  [SKIP] Campaign {campaign.uid}: {campaign.target_segment.name} "
                        f"(target reach achieved: {impressions_won}/{target_reach})"
                    )
                    continue

                remaining_reach = max(1, target_reach - impressions_won)

                # Time-related quantities
                current_day = self.get_current_day()
                # Treat campaign start/end days as inclusive
                total_duration = max(1, campaign.end_day - campaign.start_day + 1)
                days_elapsed = max(0, current_day - campaign.start_day)
                days_left = max(1, campaign.end_day - current_day + 1)

                completion_frac = impressions_won / float(target_reach)
                time_frac = days_elapsed / float(total_duration) if total_duration > 0 else 1.0

                if time_frac > 0:
                    progress_ratio = completion_frac / time_frac
                else:
                    progress_ratio = 1.0

                # Urgency: behind schedule => bid more; ahead => bid less
                if days_left <= 1 and impressions_won < target_reach:
                    urgency = 2.0  # final-day push
                    urgency_label = "FINAL DAY"
                elif progress_ratio < 0.8:
                    urgency = 1.5  # clearly behind
                    urgency_label = "BEHIND"
                elif progress_ratio > 1.2:
                    urgency = 0.7  # comfortably ahead
                    urgency_label = "AHEAD"
                else:
                    urgency = 1.0  # roughly on track
                    urgency_label = "ON_TRACK"

                # Base bid ~ remaining budget per remaining impression
                base_bid = remaining_budget / float(remaining_reach)

                # Apply urgency
                bid_per_item = base_bid * urgency

                # Clamp bids to a reasonable range.
                # Tier1 uses [0.1, 1.0]; we allow a bit more headroom.
                bid_per_item = max(0.10, min(1.25, bid_per_item))

                # Daily spending limit to avoid blowing all budget in one burst
                if days_left > 1:
                    per_day_budget = remaining_budget / float(days_left)
                    bid_limit = min(remaining_budget, per_day_budget * 1.5)
                else:
                    # Last day: it's now or never for this campaign
                    bid_limit = remaining_budget

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

                cid = campaign.uid
                if cid not in self.ad_bid_history:
                    self.ad_bid_history[cid] = []
                self.ad_bid_history[cid].append({
                    "day": current_day,
                    "bid_per_item": float(bid_per_item),
                    "bid_limit": float(bid_limit),
                    "remaining_reach": int(remaining_reach),
                    "remaining_budget": float(remaining_budget),
                    "base_bid": float(base_bid),
                    "urgency": float(urgency),
                    "urgency_label": urgency_label,
                })

                # Debug logs
                impressions_needed_per_day = remaining_reach / float(days_left) if days_left > 0 else 0.0
                completion_pct = completion_frac * 100.0
                self._log(f"  [BID] Campaign {campaign.uid}: segment={campaign.target_segment.name}")
                self._log(
                    f"        Bid/item: ${bid_per_item:.4f}, Limit: ${bid_limit:.2f} "
                    f"(base={base_bid:.4f}, urgency={urgency:.2f} [{urgency_label}])"
                )
                self._log(
                    f"        Progress: {impressions_won}/{campaign.reach} impressions "
                    f"({completion_pct:.1f}%), cost={cost_so_far:.2f}/{total_budget:.2f}"
                )
                self._log(
                    f"        Days left: {days_left}, "
                    f"Need: {impressions_needed_per_day:.1f} impressions/day to hit target {target_reach}"
                )

            except Exception as e:
                self._log(f"  [ERROR] Campaign {campaign.uid if 'campaign' in locals() and campaign else '?'} "
                          f"processing error: {e}")
                continue

        self._log(f"Total ad bid bundles created: {len(bundles)}\n")
        return bundles

    # -------------------------------------------------------------------------
    # CAMPAIGN BIDDING: reverse auctions
    # -------------------------------------------------------------------------

    def get_campaign_bids(self, campaigns_for_auction: Set[Campaign]) -> Dict[Campaign, float]:
        """
        Bid in the reverse auction based on:
        - Difficulty (fraction of available impressions needed).
        - Overlap with existing campaigns.
        - Scale (reach / duration).

        Also *skip* campaigns that are basically impossible given reach / duration.
        """
        bids: Dict[Campaign, float] = {}

        current_day = self.get_current_day()
        active_campaigns = list(self.get_active_campaigns())
        active_count = len(active_campaigns)

        self._log(f"\n--- CAMPAIGN AUCTION (Day {current_day}) ---")
        self._log(f"Campaigns available for bidding: {len(campaigns_for_auction)}")
        self._log(f"Currently active campaigns: {active_count}/{self.max_active_campaigns}")

        if active_count >= self.max_active_campaigns:
            self._log(f"[SKIP ALL] Already at max capacity ({self.max_active_campaigns} active campaigns)")
            self._log(f"{'=' * 60}\n")
            return bids

        max_new_campaigns = self.max_active_campaigns - active_count
        quality_score = self.get_quality_score() or 1.0
        self._log(f"Quality score entering campaign auction: {quality_score:.4f}")

        candidates = []

        for campaign in campaigns_for_auction:
            try:
                if campaign is None:
                    continue

                # Duration inclusive
                duration = campaign.end_day - campaign.start_day + 1
                if duration <= 0:
                    continue

                est_users_per_day = self._estimate_segment_size(campaign.target_segment)
                total_est_imps = est_users_per_day * duration

                if total_est_imps <= 0:
                    # Completely unknown → treat as insanely hard
                    fraction_needed = float("inf")
                else:
                    fraction_needed = campaign.reach / total_est_imps

                # HARD FILTERS: skip clearly impossible campaigns
                # e.g. need more than 80% of all users in that segment
                # or 1-day campaigns with very high fraction_needed.
                if fraction_needed > 0.8 and duration == 1:
                    self._log(
                        f"  [SKIP] {campaign.target_segment.name}: reach={campaign.reach}, "
                        f"dur={duration}, fraction_needed={fraction_needed:.3f} (too hard for 1 day)"
                    )
                    continue
                if fraction_needed > 1.2:
                    self._log(
                        f"  [SKIP] {campaign.target_segment.name}: reach={campaign.reach}, "
                        f"dur={duration}, fraction_needed={fraction_needed:.3f} (>120% of segment)"
                    )
                    continue

                # Difficulty weight
                if fraction_needed <= 0.4:
                    w_difficulty = 1.3
                elif fraction_needed <= 0.7:
                    w_difficulty = 1.0
                else:
                    w_difficulty = 0.6

                # Overlap: penalize if we already have campaigns on same segment
                same_segment_active = sum(
                    1 for c in active_campaigns
                    if c.target_segment.name == campaign.target_segment.name
                )
                if same_segment_active == 0:
                    w_overlap = 1.2
                elif same_segment_active == 1:
                    w_overlap = 0.8
                else:
                    w_overlap = 0.5

                # Scale factor: reach per day
                base_scale = campaign.reach / float(duration)

                # Combine into a score
                score = base_scale * w_difficulty * w_overlap

                candidates.append({
                    "campaign": campaign,
                    "score": score,
                    "norm_score": 0.0,  # fill later
                    "duration": duration,
                    "fraction_needed": fraction_needed,
                    "same_segment_active": same_segment_active,
                    "est_users_per_day": est_users_per_day,
                })
            except Exception as e:
                self._log(f"  [ERROR] Campaign evaluation error (building candidates): {e}")
                continue

        if not candidates:
            self._log("No viable campaigns to evaluate.")
            self._log(f"{'=' * 60}\n")
            return bids

        # Normalize scores for bidding
        max_score = max(c["score"] for c in candidates)
        for c in candidates:
            c["norm_score"] = c["score"] / max_score if max_score > 0 else 0.0

        # Sort best to worst
        candidates.sort(key=lambda c: c["score"], reverse=True)

        campaigns_bid_count = 0

        for cand in candidates:
            if campaigns_bid_count >= max_new_campaigns:
                self._log(f"\n[LIMIT REACHED] Already bidding on {campaigns_bid_count} campaigns this auction")
                break

            campaign = cand["campaign"]
            try:
                norm_score = cand["norm_score"]
                duration = cand["duration"]
                fraction_needed = cand["fraction_needed"]
                same_segment_active = cand["same_segment_active"]
                est_users_per_day = cand["est_users_per_day"]

                seg_name = campaign.target_segment.name
                reach = campaign.reach

                # Base multiplier in [0.2, 0.6]
                bid_multiplier = 0.2 + 0.4 * norm_score

                # Adjust for difficulty
                if fraction_needed > 0.7:
                    bid_multiplier *= 0.9
                elif fraction_needed < 0.4:
                    bid_multiplier *= 1.05

                # Adjust for our quality score
                if quality_score > 1.1:
                    bid_multiplier *= 0.95
                elif quality_score < 0.9:
                    bid_multiplier *= 1.05

                bid_multiplier = max(0.15, min(0.7, bid_multiplier))
                bid_value = reach * bid_multiplier

                min_valid_bid = reach * 0.1
                max_valid_bid = reach * 1.0
                bid_value = max(min_valid_bid, min(bid_value, max_valid_bid))

                self._log(f"\n  Campaign Analysis:")
                self._log(f"    Segment: {seg_name}")
                self._log(f"    Reach: {reach}, Duration: {duration} days")
                self._log(f"    Est. users/day: {est_users_per_day:.1f}")
                self._log(f"    Fraction needed: {fraction_needed:.3f}")
                self._log(f"    Same-segment active campaigns: {same_segment_active}")
                self._log(f"    Score: {cand['score']:.3f}, Norm score: {norm_score:.3f}")
                self._log(f"    Bid multiplier: {bid_multiplier:.3f}x reach")
                self._log(f"    Final bid: ${bid_value:.2f}  (lower wins in reverse auction)")

                if self.is_valid_campaign_bid(campaign, bid_value):
                    bids[campaign] = bid_value
                    campaigns_bid_count += 1
                    self._log(f"    [BIDDING]")
                else:
                    self._log(f"    [INVALID BID]")

            except Exception as e:
                self._log(f"  [ERROR] Campaign evaluation error: {e}")
                continue

        self._log(f"\n{'=' * 60}")
        self._log(f"SUMMARY: Bidding on {len(bids)}/{len(campaigns_for_auction)} campaigns")
        self._log(f"{'=' * 60}\n")
        return bids

    # -------------------------------------------------------------------------
    # Logging / helpers (unchanged)
    # -------------------------------------------------------------------------

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

        self._log(f"\n{'=' * 80}")
        self._log(f"DAILY CAMPAIGN STATUS - Day {current_day}")
        self._log(f"{'=' * 80}")
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
                    self._log(
                        f"    Progress: {impressions_won:,}/{campaign.reach:,} impressions "
                        f"({completion_pct:.1f}%)"
                    )
                    self._log(
                        f"    Budget: ${cost_so_far:.2f}/${campaign.budget:.2f} "
                        f"({budget_usage_pct:.1f}%)"
                    )
                    self._log(f"    Days Remaining: {campaign.end_day - current_day}")

                    # Calculate effective CPM
                    if impressions_won > 0:
                        cpm = (cost_so_far / impressions_won) * 1000
                        self._log(f"    Effective CPM: ${cpm:.4f}")

                except Exception as e:
                    self._log(f"  Campaign {campaign.uid}: Error reading status - {e}")

            self._log(f"\n  {'─' * 76}")
            self._log(f"  TOTALS:")
            self._log(f"    Budget: ${total_spent:.2f}/${total_budget:.2f}")
            self._log(f"    Impressions: {total_impressions:,}/{total_reach_target:,}")
            if total_impressions > 0:
                avg_cpm = (total_spent / total_impressions) * 1000
                self._log(f"    Average CPM: ${avg_cpm:.4f}")

        self._log(f"{'=' * 80}\n")

    def _get_matching_segments(self, target: MarketSegment) -> Set[MarketSegment]:
        # unused rn
        matching = set()

        for segment in MarketSegment.all_segments():
            if segment.issubset(target):
                matching.add(segment)

        return matching

    def _estimate_segment_size(self, segment: MarketSegment) -> float:
        """
        Approximate expected number of users per day in a segment.

        We know the base sizes for the 8 atomic segments. For broader segments
        (e.g. 'Male_Old', 'Male', 'Old_LowIncome'), we sum all atomic segments
        whose attributes are a superset of the broad segment's attributes.
        """
        base_sizes = {
            'Male_Young_LowIncome': 1836,
            'Male_Young_HighIncome': 517,
            'Male_Old_LowIncome': 1795,
            'Male_Old_HighIncome': 808,
            'Female_Young_LowIncome': 1980,
            'Female_Young_HighIncome': 256,
            'Female_Old_LowIncome': 2401,
            'Female_Old_HighIncome': 407,
        }

        name = segment.name

        # Exact atomic segment
        if name in base_sizes:
            return float(base_sizes[name])

        seg_tokens = name.split('_')
        total = 0

        # Sum all atomic segments consistent with this broader segment
        for atomic_name, size in base_sizes.items():
            atomic_tokens = atomic_name.split('_')
            # Broad segment is a subset of atomic if all its tokens appear
            # in the atomic segment's name.
            if all(tok in atomic_tokens for tok in seg_tokens):
                total += size

        # Fallback to at least 1 to avoid division by zero
        return float(max(total, 1))


    def print_debug_summary(self):
        """Print a post-game summary of campaigns, spend, fill, and Q over time."""
        self._log("\n" + "=" * 80)
        self._log(f"POST-GAME SUMMARY for {self.name}")
        self._log("=" * 80)

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

        else:
            self._log("\nNo campaign history recorded.")

        self._log("=" * 80 + "\n")


if __name__ == "__main__":
    # Here's an opportunity to test offline against some TA agents. Just run this file to do so.
    my_agent = MyNDaysNCampaignsAgent()
    test_agents = [my_agent] + [Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)]

    # Don't change this. Adapt initialization to your environment
    simulator = AdXGameSimulator()
    simulator.run_simulation(agents=test_agents, num_simulations=1)
    my_agent.print_debug_summary()
