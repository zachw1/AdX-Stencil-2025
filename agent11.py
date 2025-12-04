from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.utils.adx.structures import Bid, Campaign, BidBundle
from typing import Set, Dict, Tuple, List
import numpy as np
import math

class HybridRLAgent(NDaysNCampaignsAgent):
    def __init__(self, name: str = "brawl_stars"):
        super().__init__()
        self.name = name

        self.num_states = 9 
        

        self.beta_grid = np.linspace(0.4, 1.4, 11, dtype=np.float32)
        
        self.Q = np.zeros((self.num_states, len(self.beta_grid)), dtype=np.float32)

        self.epsilon = 1.0       
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.985 
        self.alpha = 0.1         
        self.gamma = 0.9         

        self.last_actions: Dict[int, Tuple[int, int, int, float]] = {} 
        self.max_active_campaigns = 5

    def on_new_game(self) -> None:
        super().on_new_game()
        self.last_actions.clear()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_campaign_bids(self, campaigns_for_auction: Set[Campaign]) -> Dict[Campaign, float]:
        bids: Dict[Campaign, float] = {}
        active = self.get_active_campaigns()
        quality_factor = 1.0 / max(0.1, self.get_quality_score())

        if len(active) >= self.max_active_campaigns:
            return bids

        for c in campaigns_for_auction:
            duration = c.end_day - c.start_day
            if duration < 2 and c.reach > 800: continue 

            bid_val = c.reach * 1.1 * quality_factor
            bid_val = self.clip_campaign_bid(c, bid_val)
            
            if self.is_valid_campaign_bid(c, bid_val):
                bids[c] = bid_val
        return bids

    def get_ad_bids(self) -> Set[BidBundle]:
        bundles: Set[BidBundle] = set()
        current_day = self.get_current_day()

        self._batch_update_q_table()

        active_campaigns = list(self.get_active_campaigns())
        self.last_actions.clear()

        for c in active_campaigns:
            won = self.get_cumulative_reach(c) or 0
            spent = self.get_cumulative_cost(c) or 0.0
            R_left = max(0, c.reach - won)
            B_left = max(0.0, c.budget - spent)
            days_left = max(1, c.end_day - current_day + 1)

            if B_left <= 0 or R_left <= 0: continue


            marginal_utility = self._calculate_marginal_utility(won, c.reach)
            true_value_per_impression = marginal_utility * c.budget
            
            needed_per_day = R_left / days_left
            state_idx = self._get_state_index(days_left, needed_per_day, c.reach)

            if np.random.rand() < self.epsilon:
                beta_idx = np.random.randint(len(self.beta_grid))
            else:
                beta_idx = int(np.argmax(self.Q[state_idx]))
            
            shading_factor = float(self.beta_grid[beta_idx])


            bid_price = true_value_per_impression * shading_factor
            
            bid_price = max(0.01, min(bid_price, B_left, 5.0))

            self.last_actions[c.uid] = (state_idx, beta_idx, won, spent) 

            bid = Bid(
                bidder=self,
                auction_item=c.target_segment,
                bid_per_item=bid_price,
                bid_limit=B_left,
            )
            bundle = BidBundle(
                campaign_id=c.uid,
                limit=B_left,
                bid_entries={bid},
            )
            bundles.add(bundle)

        return bundles

    def _calculate_marginal_utility(self, x: int, R: int) -> float:
        if R <= 0: return 0.0
        
        a = 4.08577
        b = 3.08577
        
        # u = a(x/R) - b
        u = a * (x / R) - b
        
        denominator = R * (1 + u**2)
        return 2.0 / denominator

    def _get_state_index(self, days_left, needed_per_day, total_reach) -> int:
        if days_left <= 1: t = 0
        elif days_left <= 3: t = 1
        else: t = 2

        ratio = needed_per_day / max(1, total_reach)
        if ratio < 0.1: p = 0
        elif ratio < 0.3: p = 1
        else: p = 2

        return (t * 3) + p

    def _batch_update_q_table(self):

        current_campaigns_map = {c.uid: c for c in self.get_active_campaigns()}
        current_day = self.get_current_day()

        for c_uid, (prev_state, action_idx, prev_won, prev_spent) in self.last_actions.items():
            if c_uid not in current_campaigns_map: continue 

            c = current_campaigns_map[c_uid]
            curr_won = self.get_cumulative_reach(c)
            curr_spent = self.get_cumulative_cost(c)

            prev_val = self._calculate_effective_reach(prev_won, c.reach) * c.budget
            curr_val = self._calculate_effective_reach(curr_won, c.reach) * c.budget
            
            value_gained = curr_val - prev_val
            cost_incurred = curr_spent - prev_spent
            
            reward = value_gained - cost_incurred

            days_left = max(0, c.end_day - current_day + 1)
            if days_left == 0 or (c.reach - curr_won) <= 0:
                max_future_q = 0.0
            else:
                R_left = max(1, c.reach - curr_won)
                needed = R_left / days_left
                next_idx = self._get_state_index(days_left, needed, c.reach)
                max_future_q = np.max(self.Q[next_idx])

            curr_q = self.Q[prev_state, action_idx]
            self.Q[prev_state, action_idx] = curr_q + self.alpha * (
                reward + (self.gamma * max_future_q) - curr_q
            )

    def _calculate_effective_reach(self, x: int, R: int) -> float:
        if R == 0: return 0.0
        a = 4.08577
        b = 3.08577
        term1 = math.atan(a * (x / R) - b)
        term2 = math.atan(-b)
        return (2.0 / a) * (term1 - term2)

    # def print_debug_summary(self):
    #      print(f"[{self.name}] Eps: {self.epsilon:.2f} | MaxQ: {np.max(self.Q):.2f}")