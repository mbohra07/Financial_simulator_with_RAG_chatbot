from functions.economic_context import EconomicEnvironment, simulate_monthly_market
from functions.monthly_simulation import deduplicate_and_save, assign_persona, generate_monthly_reflection_report
import os
import json
from litellm import RateLimitError, completion
from crew import FinancialCrew
import time
from crewai import Crew
import traceback
import streamlit as st
import requests
import httpx
import asyncio
import re
from functions.kickoff_functions import check_tool_status, execute_task_with_retries, save_task_outputs
from functions.task_functions import *
from datetime import datetime

# *******************************************Functions to sequentially and hierarchically run my crew workflow************************

def kickoff_sequential(self, inputs, sleep_between_calls=15, max_retries=20):
    print("🚀 Starting Sequential Crew Execution...")
    results = []

    month_number = inputs.get('Month')
    user_id = inputs.get('user_id')

    for task in self.tasks:
        print(f"\n🟢 Executing task: {task.name} for agent: {task.agent.role}")
        
        # Check tool status
        check_tool_status(task)
        
        # Execute task with retries
        parsed_result = execute_task_with_retries(task, inputs, max_retries)
        
        # Store result
        results.append({
            "task_name": task.name,
            "result": parsed_result
        })

        # Save output files
        if hasattr(task, 'output_file') and task.output_file:
            save_task_outputs(task, parsed_result, user_id, month_number)

        # Sleep between tasks
        time.sleep(sleep_between_calls)

    print("🎉 All tasks completed sequentially!")
    return results


# Patch the method into your Crew instance
Crew.kickoff_sequential = kickoff_sequential

# *****************************************************Simulation for my Crew workflow****************************************************

def run_simulation_with_retries(inputs, custom_agents=None, custom_tasks=None, max_attempts=3):
    hashable_inputs = json.dumps(inputs, sort_keys=True)
    customized_agents = custom_agents
    customized_tasks = custom_tasks
    for attempt in range(max_attempts):
        try:
            result = FinancialCrew().flexible_crew(
                input_data=hashable_inputs,
                agent_overrides=customized_agents,
                task_overrides=customized_tasks
            ).kickoff_sequential(inputs=inputs)
            return result
        except RateLimitError:
            st.warning(f"Rate limit hit. Retrying in 10 seconds... (Attempt {attempt + 1}/{max_attempts})")
            time.sleep(10)
        except Exception as e:
            print("Full Traceback:")
            traceback.print_exc()
            st.error(f"An unexpected error occurred: {e}")
            break
    return None



def simulate_timeline(n_months, simulation_unit, user_inputs, task_id=None):
    global simulation_tasks
    
    import os
    import json
    import asyncio
    import httpx
    from datetime import datetime

    for month in range(1, n_months + 1):
        eco_env = EconomicEnvironment(unit=simulation_unit)
        eco_env.simulate_step()
        eco_context = eco_env.get_context()
        market_snapshot, market_context_summary = simulate_monthly_market()
        user_inputs["market_context"] = market_context_summary 
        user_inputs['inflation'] = eco_context['inflation_rate']
        user_inputs['interest_rate'] = eco_context['interest_rate']
        user_inputs['cost_of_living_index'] = eco_context['cost_of_living_index']
        user_name = user_inputs['user_name']
        user_id = user_inputs['user_id']
        user_inputs['Month'] = month

        print(f"\n🚀 Simulating Month {month}")
        result = run_simulation_with_retries(inputs=user_inputs)
        assign_persona(user_name, month)
        generate_monthly_reflection_report(user_name, month)
        
        """output_dir = 'output'

        # ✅ Collect simulation result files
        file_keys = [
            "behavior_tracker", "discipline_report", "financial_strategy",
            "goal_status", "karmic_tracker", "simulated_cashflow"
        ]
        simulation_outputs = {}
        try:
            # ✅ Load monthly simulation result files
            for key in file_keys:
                filename = f"{user_id}_{key}_simulation_{month}.json"
                filepath = os.path.join(output_dir, filename)
                if os.path.exists(filepath):
                    with open(filepath, "r") as f:
                        simulation_outputs[key] = json.load(f)
                else:
                    simulation_outputs[key] = f"{filename} not found"
 
            # ✅ Load persona history
            persona_path = os.path.join("data", "persona_history.json")
            if os.path.exists(persona_path):
                with open(persona_path, "r") as f:
                    simulation_outputs["persona_history"] = json.load(f)
            else:
                simulation_outputs["persona_history"] = "persona_history.json not found"
            # ✅ Load reflection report
            reflection_filename = f"reflection_month_{month}.json"
            reflection_path = os.path.join("monthly_output", reflection_filename)
            if os.path.exists(reflection_path):
                with open(reflection_path, "r") as f:
                    simulation_outputs["reflection_report"] = json.load(f)
            else:
                simulation_outputs["reflection_report"] = f"{reflection_filename} not found"

            # Add metadata
            simulation_outputs["metadata"] = {
                "user_id": user_id,
                "user_name": user_name,
                "month": month,
                "simulation_unit": simulation_unit,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as read_err:
            print(f"❌ Error reading output files: {read_err}")
            simulation_outputs = {
                "error": str(read_err),
                "user_id": user_id,
                "user_name": user_name,
                "month": month
            }

        # ✅ Notify frontend with GET request
        try:
            async def notify_frontend():
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(
                            "http://192.168.0.109:8000/get-simulation-result",  
                            params={
                                "user_name": user_name,
                                "month": month,
                                "result": json.dumps(simulation_outputs)
                            },
                            timeout=10.0  # Add timeout to prevent hanging
                        )
                        return response.status_code
                except httpx.RequestError as e:
                    print(f"❌ Request error when notifying frontend: {e}")
                    return None
                except Exception as e:
                    print(f"❌ Unexpected error when notifying frontend: {e}")
                    return None

            # Run the async function with proper error handling
            try:
                status_code = asyncio.run(notify_frontend())
                if status_code:
                    print(f"✅ Notified frontend for Month {month} (Status: {status_code})")
                else:
                    print(f"⚠️ Could not notify frontend for Month {month}")
            except RuntimeError as e:
                # Handle "Event loop is already running" error
                if "already running" in str(e):
                    print(f"⚠️ Could not notify frontend (event loop issue): {e}")
                else:
                    raise
                
        except Exception as e:
            print(f"❌ Failed to notify frontend in Month {month}: {e}")"""
            
    print("\n🎉 All months simulated successfully!")
    return True
