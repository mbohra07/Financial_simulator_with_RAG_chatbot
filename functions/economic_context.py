# economic_context.py

import random
import wbgapi as wb
import requests

class RealDataProvider:
    def __init__(self):
        self.context = {
            "inflation_rate": 0.05,
            "interest_rate": 0.7,
            "cost_of_living_index": 1.0
        }

    def get_economic_data(self):

        try:
            inflation = wb.data.DataFrame('FP.CPI.TOTL.ZG', 'IN', mrv=1).iloc[0, 0]
            self.context['inflation_rate'] = round(inflation / 100, 4)
        except Exception as e:
            print(f"Inflation fetch failed: {e}")

        try:
            interest = wb.data.DataFrame('FR.INR.LEND', 'IN', mrv=1).iloc[0, 0]
            self.context['interest_rate'] = round(interest / 100, 4)
        except Exception as e:
            print(f"Interest fetch failed: {e}")

        return self.context.copy()
    
class SimulatedDataProvider:
    def __init__(self):
        self.context = {
            "inflation_rate": 0.05,
            "interest_rate": 0.7,
            "cost_of_living_index": 1.0
        }

    def simulate_step(self, unit):
        if unit == "Days":
            self.context["inflation_rate"] *= 1 + random.uniform(-0.0002, 0.0002)
            self.context["interest_rate"] += random.uniform(-0.005, 0.005)
            self.context["cost_of_living_index"] += random.uniform(-0.001, 0.001)
        elif unit == "Months":
            self.context["inflation_rate"] *= 1 + random.uniform(-0.005, 0.005)
            self.context["interest_rate"] += random.uniform(-0.02, 0.02)
            self.context["cost_of_living_index"] += random.uniform(-0.01, 0.01)

    def get_economic_data(self):
        return self.context.copy()
    
class EconomicEnvironment:
    def __init__(self, unit="Months", provider_type="simulated"):
        self.unit = unit
        if provider_type == "simulated":
            self.provider = SimulatedDataProvider()
        elif provider_type == "real":
            self.provider = RealDataProvider()
        else:
            raise ValueError("Unsupported provider type.")

    def simulate_step(self):
        if hasattr(self.provider, "simulate_step"):
            self.provider.simulate_step(self.unit)

    def get_context(self):
        return self.provider.get_economic_data()
    
def simulate_monthly_market():
    market_statuses = ["bullish", "bearish", "volatile"]
    market_categories = ["crypto", "stocks", "commodities"]
    
    def get_weights(category):
        return {
            "crypto": [0.3, 0.2, 0.5],
            "stocks": [0.4, 0.4, 0.2],
            "commodities": [0.35, 0.35, 0.3]
        }.get(category, [0.33, 0.33, 0.34])
    
    snapshot = {
        category: random.choices(market_statuses, weights=get_weights(category), k=1)[0]
        for category in market_categories
    }
    
    # Generate LLM-friendly summary
    summary = (
        f"Market conditions this month: "
        f"Crypto is {snapshot['crypto']}, Stocks are {snapshot['stocks']}, Commodities are {snapshot['commodities']}. "
        f"Adjust investment returns, job security, and consumer sentiment accordingly."
    )
    
    return snapshot, summary

