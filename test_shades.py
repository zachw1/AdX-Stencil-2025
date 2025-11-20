"""
Test different shade values to find optimal parameter.
Runs multiple agents with varying shades and sees which performs best.
"""
from agent4 import MyNDaysNCampaignsAgent
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.local_games.adx_arena import AdXGameSimulator
from typing import Dict, List
import numpy as np


class ShadedAgent(MyNDaysNCampaignsAgent):
    """Agent with configurable shade parameter."""
    
    def __init__(self, shade_value: float, name: str = None):
        super().__init__()
        self.base_shade = shade_value  # Store the shade to add
        self.name = name or f"Shade_{shade_value:.2f}"
    
    def get_ad_bids(self):
        """Override to use custom shade value."""
        from agt_server.agents.utils.adx.structures import Bid, BidBundle
        
        bundles = set()
        current_day = self.get_current_day()
        
        # Track quality score
        quality_score = self.get_quality_score() or 1.0
        if not self.daily_quality_scores or self.daily_quality_scores[-1][0] != current_day:
            self.daily_quality_scores.append((current_day, quality_score))

        for campaign in self.get_active_campaigns():
            impressions_won = self.get_cumulative_reach(campaign) or 0
            cost_so_far = self.get_cumulative_cost(campaign) or 0.0
            
            remaining_budget = max(0.0, campaign.budget - cost_so_far)
            if remaining_budget <= 0.0:
                continue
            
            if impressions_won >= campaign.reach:
                continue
            
            remaining_reach = campaign.reach - impressions_won
            
            # Calculate average value
            current_rho = self.effective_reach(impressions_won, campaign.reach)
            target_rho = self.effective_reach(campaign.reach, campaign.reach)
            delta_rho = target_rho - current_rho
            avg_value_per_impression = (delta_rho * campaign.budget) / remaining_reach if remaining_reach > 0 else 0
            
            # Calculate marginal value at future position
            segment_name = campaign.target_segment.name
            if segment_name in self.segment_sizes:
                reach_target = self.segment_sizes[segment_name]
            else:
                reach_target = 1000  # Default
                
            marginal_rho = self._marginal_effective_reach(impressions_won, campaign.reach, reach_target)
            marginal_value_per_impression = marginal_rho * campaign.budget
            
            # Calculate progress shade
            if avg_value_per_impression > 0:
                progress_shade = marginal_value_per_impression / avg_value_per_impression
            else:
                progress_shade = 1.0
            
            # USE CUSTOM SHADE VALUE (this is what we're testing!)
            bid_per_item = avg_value_per_impression * (progress_shade + self.base_shade)
            
            bid_limit = remaining_budget
            bid_per_item = min(bid_per_item, bid_limit)
            
            if bid_per_item <= 0 or bid_limit <= 0:
                continue
            
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
            
            # Track for logging
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
        
        return bundles


def run_shade_experiment(shade_values: List[float], num_simulations: int = 10):
    """
    Run simulations with different shade values and report results.
    
    Args:
        shade_values: List of shade values to test (e.g., [0.2, 0.3, 0.4, 0.5])
        num_simulations: Number of games to run for averaging
    """
    
    print("\n" + "=" * 120)
    print(f"SHADE PARAMETER OPTIMIZATION - Testing {len(shade_values)} shade values over {num_simulations} simulations")
    print("=" * 120)
    
    # Create agents with different shades
    test_agents = []
    for shade in shade_values:
        agent = ShadedAgent(shade_value=shade)
        test_agents.append(agent)
    
    # Add some baseline Tier1 agents for comparison
    test_agents += [Tier1NDaysNCampaignsAgent(name=f"Tier1_{i}") for i in range(3)]
    
    # Run simulations
    print(f"\nRunning {num_simulations} simulations with {len(test_agents)} agents...")
    print(f"Shade values being tested: {shade_values}")
    
    simulator = AdXGameSimulator()
    total_profits = {agent.name: 0.0 for agent in test_agents}
    
    for sim in range(num_simulations):
        print(f"\n  Simulation {sim + 1}/{num_simulations}...", end=" ", flush=True)
        results = simulator.run_simulation(agents=test_agents, num_simulations=1)
        
        # Accumulate profits
        for agent in test_agents:
            if agent.name in results:
                total_profits[agent.name] += results[agent.name]
        
        print("✓")
    
    # Calculate average profits
    avg_profits = {name: profit / num_simulations for name, profit in total_profits.items()}
    
    # Separate shaded agents from baseline
    shaded_results = []
    baseline_results = []
    
    for agent in test_agents:
        avg_profit = avg_profits[agent.name]
        if isinstance(agent, ShadedAgent):
            shaded_results.append((agent.base_shade, avg_profit))  # Fixed: use base_shade
        else:
            baseline_results.append((agent.name, avg_profit))
    
    # Sort by profit
    shaded_results.sort(key=lambda x: x[1], reverse=True)
    baseline_results.sort(key=lambda x: x[1], reverse=True)
    
    # Print results
    print("\n" + "=" * 120)
    print("RESULTS: Shaded Agents")
    print("=" * 120)
    print(f"\n{'Rank':<6} {'Shade':<10} {'Avg Profit':<15} {'Performance'}")
    print("-" * 60)
    
    for rank, (shade, profit) in enumerate(shaded_results, 1):
        star = "⭐ BEST!" if rank == 1 else ""
        print(f"{rank:<6} {shade:<10.2f} ${profit:<14.2f} {star}")
    
    print("\n" + "=" * 120)
    print("BASELINE: Tier1 Agents")
    print("=" * 120)
    print(f"\n{'Agent':<20} {'Avg Profit':<15}")
    print("-" * 40)
    
    for name, profit in baseline_results:
        print(f"{name:<20} ${profit:<14.2f}")
    
    # Statistical analysis
    print("\n" + "=" * 120)
    print("STATISTICAL ANALYSIS")
    print("=" * 120)
    
    profits_only = [p for _, p in shaded_results]
    shades_only = [s for s, _ in shaded_results]
    
    best_shade, best_profit = shaded_results[0]
    worst_shade, worst_profit = shaded_results[-1]
    
    print(f"\n✅ BEST  Shade: {best_shade:.2f} → Avg Profit: ${best_profit:.2f}")
    print(f"❌ WORST Shade: {worst_shade:.2f} → Avg Profit: ${worst_profit:.2f}")
    print(f"📊 Range: ${(best_profit - worst_profit):.2f} difference")
    print(f"📊 Mean Profit: ${np.mean(profits_only):.2f}")
    print(f"📊 Std Dev: ${np.std(profits_only):.2f}")
    
    # Find optimal region
    print("\n💡 RECOMMENDATION:")
    if len(shaded_results) >= 3:
        top_3_shades = [s for s, _ in shaded_results[:3]]
        avg_top_3 = np.mean(top_3_shades)
        print(f"   Top 3 shades: {top_3_shades}")
        print(f"   Average of top 3: {avg_top_3:.3f}")
        print(f"   Suggested shade: {avg_top_3:.3f}")
    else:
        print(f"   Use shade = {best_shade:.2f}")
    
    print("\n" + "=" * 120 + "\n")
    
    return shaded_results


if __name__ == "__main__":
    # Test a range of shade values
    # Start with coarse grid
    print("\n🔬 EXPERIMENT 1: Coarse Grid Search")
    print("Testing shades from 0.0 to 1.0 in steps of 0.1")
    
    coarse_shades = np.linspace(0.0, 1.0, 11).tolist()  # [0.0, 0.1, 0.2, ..., 1.0]
    coarse_results = run_shade_experiment(coarse_shades, num_simulations=5)
    
    # Fine-tune around best result
    best_shade = coarse_results[0][0]
    
    print("\n🔬 EXPERIMENT 2: Fine Grid Search")
    print(f"Fine-tuning around best shade: {best_shade:.2f}")
    
    # Test ±0.15 around best with finer granularity
    fine_min = max(0.0, best_shade - 0.15)
    fine_max = min(1.0, best_shade + 0.15)
    fine_shades = np.linspace(fine_min, fine_max, 7).tolist()
    fine_results = run_shade_experiment(fine_shades, num_simulations=10)
    
    # Final recommendation
    optimal_shade = fine_results[0][0]
    optimal_profit = fine_results[0][1]
    
    print("\n" + "=" * 120)
    print("🎯 FINAL RECOMMENDATION")
    print("=" * 120)
    print(f"\n✨ OPTIMAL SHADE: {optimal_shade:.3f}")
    print(f"💰 Expected Avg Profit: ${optimal_profit:.2f}")
    print(f"\nUpdate agent4.py line with:")
    print(f"    bid_per_item = avg_value_per_impression * (progress_shade + {optimal_shade:.3f})")
    print("\n" + "=" * 120 + "\n")

