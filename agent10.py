from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.local_games.adx_arena import AdXGameSimulator
from agt_server.agents.utils.adx.structures import Bid, Campaign, BidBundle
from typing import Set, Dict, Tuple, List
import numpy as np


class BigBuddyNDaysNCampaignsAgent(NDaysNCampaignsAgent):
    """
    RL-based AdX agent.

    Uses Q-learning to tune a β multiplier on CPM (base price per impression = B_left / R_left).

    - State: urgency bucket based on days_left for a campaign.
        bucket 0: very urgent (0–1 days left)
        bucket 1: medium (2–3 days left)
        bucket 2: low urgency (>= 4 days left)

    - Action: index into β grid, e.g. β in [0.5, ..., 1.5].

    - Reward: Δ total profit from previous day = profit_t - profit_{t-1}.
      Every campaign's (bucket, β) chosen on day t gets updated with this same reward at day t+1.

    Campaign auctions:
      - Simple heuristic: skip 1-day campaigns, bid aggressively but within [0.1R, R].
      - Limit how many concurrent campaigns we try to manage.
    """

    def __init__(self, name: str = "RL-Beta"):
        super().__init__()
        self.name = name

        # RL parameters
        self.beta_grid = np.linspace(0.5, 1.5, 11, dtype=np.float32)  # β ∈ {0.5, 0.6, ..., 1.5}
        self.num_buckets = 3                                          # urgency buckets
        self.Q = np.zeros((self.num_buckets, len(self.beta_grid)), dtype=np.float32)

        self.eps = 0.1    # ε-greedy exploration
        self.alpha = 0.2  # learning rate

        # Game-level tracking
        self.prev_profit: float = 0.0
        # For each campaign uid, store (bucket_idx, beta_idx) chosen today
        self.last_actions: Dict[int, Tuple[int, int]] = {}

        # Optional: track quality score by day
        self.daily_quality_scores: List[Tuple[int, float]] = []

        # Limit how many campaigns we actively pursue
        self.max_active_campaigns = 5

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_new_game(self) -> None:
        """
        Called at the start of each game.
        We reset per-game state but keep the Q-table so that learning
        carries over across games.
        """
        super().on_new_game()
        self.prev_profit = 0.0
        self.last_actions.clear()
        self.daily_quality_scores.clear()

    # ------------------------------------------------------------------
    # Campaign auctions (reverse second-price auctions for budgets)
    # ------------------------------------------------------------------

    def get_campaign_bids(self, campaigns_for_auction: Set[Campaign]) -> Dict[Campaign, float]:
        """
        Return bids for campaign auctions.

        - Skip 1-day campaigns (very hard to fulfill).
        - If we already have many active campaigns, be pickier.
        - Bid aggressive but valid budgets in [0.1R, R] via clip_campaign_bid.
        """
        bids: Dict[Campaign, float] = {}

        active = self.get_active_campaigns()
        active_count = len(active)

        # Already at capacity? Don't bid for more campaigns.
        if active_count >= self.max_active_campaigns:
            return bids

        for c in campaigns_for_auction:
            if c is None:
                continue

            duration = c.end_day - c.start_day
            if duration < 2:
                # 1-day campaigns are too risky: little time to complete.
                continue

            # If nearly at capacity, only take larger campaigns (heuristic).
            if active_count >= self.max_active_campaigns - 1 and c.reach < 800:
                continue

            # Base bid proportional to reach.
            # Remember: this is a reverse auction, lower bids win,
            # but budget will be set based on second-lowest effective bid.
            raw_bid = 0.9 * c.reach  # fairly aggressive
            bid_val = self.clip_campaign_bid(c, raw_bid)

            if self.is_valid_campaign_bid(c, bid_val):
                bids[c] = bid_val

        return bids

    # ------------------------------------------------------------------
    # Ad auctions (repeated user-level auctions; RL happens here)
    # ------------------------------------------------------------------

    def get_ad_bids(self) -> Set[BidBundle]:
        """
        Called once per day before simulating ad auctions.

        Steps:
        1. Compute reward as Δ profit since yesterday.
        2. Use that reward to update Q-values for all (bucket, β) chosen yesterday.
        3. For each active campaign, pick an urgency bucket + β index via ε-greedy,
           convert that into a per-impression price, and emit a BidBundle with a
           campaign-level spending limit (the remaining budget).
        """
        bundles: Set[BidBundle] = set()
        current_day = self.get_current_day()

        # ---- 1) RL update from previous day --------------------------------
        reward = self._compute_daily_reward()
        self._update_Q_from_last_actions(reward)

        # ---- 2) Track quality score history (optional, for debugging) -------
        q = self.get_quality_score() or 1.0
        if not self.daily_quality_scores or self.daily_quality_scores[-1][0] != current_day:
            self.daily_quality_scores.append((current_day, q))

        # ---- 3) Choose ad bids for each active campaign ---------------------
        active_campaigns = list(self.get_active_campaigns())

        # If we somehow have more active campaigns than our desired max,
        # focus on the ones with the largest remaining budget.
        if len(active_campaigns) > self.max_active_campaigns:
            active_campaigns.sort(
                key=lambda c: (c.budget - (self.get_cumulative_cost(c) or 0.0)),
                reverse=True,
            )
            active_campaigns = active_campaigns[: self.max_active_campaigns]

        # Clear previous choices; we'll store today's choices here.
        self.last_actions.clear()

        for c in active_campaigns:
            impressions_won = self.get_cumulative_reach(c) or 0
            cost_so_far = self.get_cumulative_cost(c) or 0.0

            R_left = max(0, c.reach - impressions_won)
            B_left = max(0.0, c.budget - cost_so_far)

            # Nothing left to do?
            if R_left <= 0 or B_left <= 0.0:
                continue

            # Urgency bucket based on days remaining in campaign.
            days_left = max(0, c.end_day - current_day)
            bucket = self._urgency_bucket(days_left)

            # ε-greedy selection of β index.
            if np.random.rand() < self.eps:
                beta_idx = np.random.randint(len(self.beta_grid))  # explore
            else:
                beta_idx = int(np.argmax(self.Q[bucket]))          # exploit

            beta = float(self.beta_grid[beta_idx])

            # Base CPM: how much budget we have left per remaining impression.
            base_cpm = B_left / max(R_left, 1)

            # RL-adjusted price per impression.
            price = beta * base_cpm

            # Safety: ensure price and limit are reasonable and positive.
            price = max(0.01, min(price, B_left))   # at least > 0, at most remaining budget
            price = min(price, 5.0)                 # optional global max cap

            bid_limit = B_left  # campaign-level spending cap for the day

            bid = Bid(
                bidder=self,
                auction_item=c.target_segment,
                bid_per_item=price,
                bid_limit=bid_limit,
            )
            bundle = BidBundle(
                campaign_id=c.uid,
                limit=bid_limit,      # campaign spending limit (across segments)
                bid_entries={bid},    # we only bid on the campaign's exact segment
            )
            bundles.add(bundle)

            # Store today's (bucket, β) choice for RL update tomorrow.
            self.last_actions[c.uid] = (bucket, beta_idx)

        return bundles

    # ------------------------------------------------------------------
    # RL helpers
    # ------------------------------------------------------------------

    def _compute_daily_reward(self) -> float:
        """
        Reward = change in cumulative profit since last day.
        The simulator updates self.profit once per day (after campaigns end).
        """
        current_profit = self.get_cumulative_profit()
        reward = current_profit - self.prev_profit
        self.prev_profit = current_profit
        return float(reward)

    def _update_Q_from_last_actions(self, reward: float) -> None:
        """
        For every (bucket, β_idx) we used yesterday (one per campaign),
        update Q[bucket, β_idx] with the same reward.

        This is a coarse credit assignment: we don't know which exact
        impression or campaign produced the profit, but it's a simple and
        stable signal for learning which β's work better in which urgency.
        """
        if not self.last_actions:
            return
        for (bucket, beta_idx) in self.last_actions.values():
            old_val = self.Q[bucket, beta_idx]
            self.Q[bucket, beta_idx] = old_val + self.alpha * (reward - old_val)

    def _urgency_bucket(self, days_left: int) -> int:
        """
        Map days_left to one of three urgency buckets.

        0: very urgent (0–1 days left)
        1: medium urgency (2–3 days left)
        2: low urgency (>= 4 days left)
        """
        if days_left <= 1:
            return 0
        elif days_left <= 3:
            return 1
        else:
            return 2

    # ------------------------------------------------------------------
    # Optional: debugging / summary
    # ------------------------------------------------------------------

    def print_debug_summary(self):
        """Prints a simple summary of the Q-table and quality scores."""
        print("\n" + "=" * 80)
        print(f"RL AGENT SUMMARY for {self.name}")
        print("=" * 80)

        if self.daily_quality_scores:
            print("\nQuality score by day:")
            for day, q in self.daily_quality_scores:
                print(f"  Day {day}: Q = {q:.4f}")
        else:
            print("\nNo quality score history recorded.")

        print("\nQ-table (urgency_bucket x beta_index):")
        for b in range(self.num_buckets):
            row = " ".join(f"{v:7.3f}" for v in self.Q[b])
            print(f"  Bucket {b}: {row}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    # Test offline against Tier1 agents.
    my_agent = BigBuddyNDaysNCampaignsAgent()
    test_agents = [my_agent] + [
        Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)
    ]

    simulator = AdXGameSimulator()
    simulator.run_simulation(agents=test_agents, num_simulations=50)
    my_agent.print_debug_summary()
