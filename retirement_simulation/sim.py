import numpy as np
import re
import json
import sys
from dataclasses import dataclass
from typing import Dict, List

# --- UTILITIES ---

def load_jsonc(file_path: str) -> Dict:
    with open(file_path, 'r') as f:
        content = f.read()
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    content = re.sub(r'//.*', '', content)
    return json.loads(content)

@dataclass
class Scenario:
    name: str
    avg_return: float
    vol: float
    raise_rate: float
    inflation: float

# --- VECTORIZED ENGINE ---

def run_sim(n_sims: int, plan: Dict, scen: Scenario):
    curr_age = plan["CURRENT AGE"]
    ret_age = plan["RETIREMENT"]["RETIREMENT AGE"]
    death_age = plan["RETIREMENT"]["DEATH AGE"]
    years = death_age - curr_age
    
    # Initialize Portfolios: Shape (n_sims,)
    portfolios = np.full(n_sims, float(plan["PORTFOLIO"]["CURRENT"]))
    skips = np.full(n_sims, int(plan["RETIREMENT"]["EMERGENCY SAVINGS"]))
    
    for i, year_idx in enumerate(np.arange(1, years+1)):
        age = curr_age + year_idx
        
        # 1. Generate Random Returns for all sims at once
        market_returns = np.random.normal(scen.avg_return, scen.vol, n_sims)
        portfolios *= (1 + market_returns)
        
        if age <= ret_age:
            # Phase: Accumulation (Vectorized addition)
            contribution = (plan["PORTFOLIO"]["MONTHLY DEPOSIT"] * 12) * ((1 + scen.raise_rate) ** year_idx)
            portfolios += contribution
        else:
            # Phase: Retirement 
            inf_spend = plan["RETIREMENT"]["ANNUAL SPEND"] * ((1 + scen.inflation) ** year_idx)
            inf_ss = plan["RETIREMENT"]["SOCIAL SECURITY"] * ((1 + scen.inflation) ** year_idx)
            inf_div = plan["PORTFOLIO"]["ANNUAL DIVIDENDS"] * ((1 + scen.inflation) ** year_idx)
            net_withdrawal = inf_spend - inf_ss - inf_div
            
            # Logic: Skip withdrawal if market is down AND we have skips left
            should_withdraw = (market_returns > 0) | (skips <= 0)
            
            portfolios[should_withdraw] -= net_withdrawal
            
            # Decrement skips for those who used them (market < 0 and skips > 0)
            used_skip = (~should_withdraw) & (portfolios > 0)
            skips[used_skip] -= 1
            
        # Floor portfolios at 0
        portfolios = np.maximum(portfolios, 0)

    success_rate = np.sum(portfolios > 0) / n_sims * 100
    median_final = np.median(portfolios)
    
    return success_rate, median_final

# --- OUTPUT ---

def main():
    if len(sys.argv) < 2: return
    plan = load_jsonc(sys.argv[1])
    
    scenarios = [
        Scenario("Expected", 0.10, 0.15, 0.03, 0.02),
        Scenario("Conservative", 0.08, 0.18, 0.02, 0.03),
        Scenario("Catastrophic", 0.06, 0.20, 0.02, 0.03)
    ]
    
    print(f"\n{'Scenario':<15} | {'Success Rate':>12} | {'Median Ending'}")
    print("-" * 50)
    
    for s in scenarios:
        rate, median = run_sim(100000, plan, s)
        print(f"{s.name:<15} | {rate:>11.2f}% | ${median:,.0f}")

if __name__ == "__main__":
    main()
