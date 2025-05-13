from crew import FinancialCrew
import json
import os
from crew_functions.economic_context import EconomicEnvironment

eco_env = EconomicEnvironment(unit="Months")

for _ in range(2):
  eco_env.simulate_step()
  context = eco_env.get_context()

  economic_context = context
  inflation = economic_context['inflation_rate']
  interest_rate = economic_context['interest_rate']
  cost_of_living_index = economic_context['cost_of_living_index']

n_iterations = 2
inputs = {
            'user_name': 'Madhuram Bohra',
            'age': 22,
            'occupation': "Student",
            'income_level': "<10,000",
            'goal': 'Save ₹50,000 for emergency fund',
            'starting_balance': 10000,
            'simulation_unit': "Months",
            'simulation_steps': 3,
            'inflation': inflation,
            'interest_rate': interest_rate,
            'cost_of_living_index': cost_of_living_index,
            'monthly_earning': 15000,
            'savings_target': 10000,
            'percentage_of_my_savings': 33,
            'monthly_expenses': 13000,
            'cashflow_context': 'There is no previous summary'
        }
filename = "your_model.pkl"


try:
    FinancialCrew().flexible_crew().train(
      n_iterations=n_iterations, 
      inputs=inputs, 
      filename=filename
    )

except Exception as e:
    raise Exception(f"An error occurred while training the crew: {e}")