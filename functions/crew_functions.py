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
from functions.task_functions import *
from datetime import datetime

# *******************************************Functions to sequentially and hierarchically run my crew workflow************************

def kickoff_sequential(self, inputs, sleep_between_calls=15, max_retries=20):
    print("🚀 Starting Sequential Crew Execution...")
    results = []

    month_number = inputs.get('Month', 1)
    user_id = inputs.get('user_id')

    for task in self.tasks:
        print(f"\n🟢 Executing task: {task.name} for agent: {task.agent.role}")

        # 🔎 Check tool status
        if hasattr(task, 'tool') and task.tool:
            try:
                tool_status = task.tool.check_status()
                if tool_status:
                    print(f"🛠️ Tool '{task.tool.name}' is working properly ✅")
                else:
                    print(f"⚠️ Tool '{task.tool.name}' is NOT responding ❌")
            except Exception as e:
                print(f"❗ Error checking tool '{task.tool.name}': {e}")
        else:
            print(f"ℹ️ No tool assigned to task '{task.name}'")

        # ➕ Load context
        context_inputs = inputs.copy()
        prev_result = None
        cashflow_data = None

        if hasattr(task, 'output_file') and task.output_file:
            output_dir = os.path.dirname(task.output_file)
            base_file_name = os.path.basename(task.output_file)
            file_name_without_ext, ext = os.path.splitext(base_file_name)

        # Determine the required context based on task dependencies
        task_context = []
        
        if task.name == "simulate_cashflow_task":
            # Only need previous simulate_cashflow and financial_strategy
            task_context.append(build_simulated_cashflow_context(month_number, user_id))
            task_context.append(build_financial_strategy_context(month_number, user_id))
        elif task.name == "discipline_tracker_task":
            # Needs context for previous simulate_cashflow and discipline_tracker
            task_context.append(build_simulated_cashflow_context(month_number, user_id))
            task_context.append(build_discipline_report_context(month_number, user_id))
        
        elif task.name == "track_goals":
            # Needs context for previous simulate_cashflow, discipline_tracker, and goal_tracker
            task_context.append(build_simulated_cashflow_context(month_number, user_id))
            task_context.append(build_discipline_report_context(month_number, user_id))
            task_context.append(build_goal_status_context(month_number, user_id))
        
        elif task.name == "behavior_tracker_task":
            # Needs context for previous simulate_cashflow and behavior_tracker
            task_context.append(build_simulated_cashflow_context(month_number, user_id))
            task_context.append(build_behavior_tracker_context(month_number, user_id))
        
        elif task.name == "karma_tracker_task":
            # Needs context for previous simulate_cashflow, behavior_tracker, and karma_tracker
            task_context.append(build_simulated_cashflow_context(month_number, user_id))
            task_context.append(build_behavior_tracker_context(month_number, user_id))
            task_context.append(build_karmic_tracker_context(month_number, user_id))
        
        elif task.name == "financial_strategy_task":
            # Needs all previous context
            task_context.append(build_simulated_cashflow_context(month_number, user_id))
            task_context.append(build_financial_strategy_context(month_number, user_id))
            task_context.append(build_discipline_report_context(month_number, user_id))
            task_context.append(build_karmic_tracker_context(month_number, user_id))
            task_context.append(build_goal_status_context(month_number, user_id))
            task_context.append(build_behavior_tracker_context(month_number, user_id))

        # Combine all context strings for the task
        context_inputs['context'] = "\n\n".join([c for c in task_context if c])

        # 🚀 Execute task
        retries = 0
        task_result = None
        parsed_result = None
        while retries < max_retries:
            task_result = task.agent.execute_task(task, context_inputs)
            try:
                if isinstance(task_result, str):
                    parsed_result = json.loads(task_result)
                else:
                    parsed_result = task_result
                print(f"✅ Finished task: {task.name}\nResult is valid JSON ✅")
                break
            except json.JSONDecodeError:
                retries += 1
                print(f"❗ Task result is not valid JSON (Attempt {retries}/{max_retries}). Retrying...")
                time.sleep(10)

        if parsed_result is None:
            print(f"❌ Failed to get valid JSON for task '{task.name}' after {max_retries} retries. Saving last result as plain text.")
            parsed_result = task_result

        results.append({
            "task_name": task.name,
            "result": parsed_result
        })

        # 💾 Write output (append mode for every task)
        if hasattr(task, 'output_file') and task.output_file:
            output_dir = os.path.dirname(task.output_file)
            monthly_output_dir = "monthly_output"
            base_file_name = os.path.basename(task.output_file)
            file_name_without_ext, ext = os.path.splitext(base_file_name)
            monthly_file_name = f"{file_name_without_ext}_simulation_{month_number}{ext}"
            dynamic_file_name = f"{user_id}_{file_name_without_ext}_simulation{ext}"
            monthly_output_path = os.path.join(monthly_output_dir, monthly_file_name)
            output_path = os.path.join(output_dir, dynamic_file_name)

            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(monthly_output_dir, exist_ok=True)

            try:
                # Load existing data if exists
                if os.path.exists(output_path):
                    with open(output_path, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                else:
                    existing_data = []

                # If parsed_result is a list, extend, else append
                if isinstance(parsed_result, list):
                    existing_data.extend(parsed_result)
                else:
                    existing_data.append(parsed_result)

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(existing_data, f, indent=4, ensure_ascii=False)

                print(f"💾 Appended result to {output_path}")
            except Exception as e:
                print(f"❗ Error appending to ouput file '{output_path}': {e}")

            try:
                # Load existing data if exists
                if os.path.exists(monthly_output_path):
                    with open(monthly_output_path, "r", encoding="utf-8") as f:
                        existing_monthly_data = json.load(f)
                else:
                    existing_monthly_data = []

                # If parsed_result is a list, extend, else append
                if isinstance(parsed_result, list):
                    existing_monthly_data.extend(parsed_result)
                else:
                    existing_monthly_data.append(parsed_result)

                with open(monthly_output_path, "w", encoding="utf-8") as f:
                    json.dump(existing_monthly_data, f, indent=4, ensure_ascii=False)

                print(f"💾 Appended result to {monthly_output_path}")
            except Exception as e:
                print(f"❗ Error appending to monthly file '{monthly_output_path}': {e}")

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
    
    previous_result = None

    for month in range(1, n_months + 1):
        eco_env = EconomicEnvironment(unit=simulation_unit)
        eco_env.simulate_step()
        context = eco_env.get_context()
        market_snapshot, market_context_summary = simulate_monthly_market()
        user_inputs["market_context"] = market_context_summary 
        user_name = user_inputs['user_name']
        user_id = user_inputs['user_id']
        user_inputs['inflation'] = context['inflation_rate']
        user_inputs['interest_rate'] = context['interest_rate']
        user_inputs['cost_of_living_index'] = context['cost_of_living_index']
        user_inputs['Month'] = month

        print(f"\n🚀 Simulating Month {month}")
        result = run_simulation_with_retries(inputs=user_inputs)
        assign_persona(user_name, month)
        generate_monthly_reflection_report(user_name, month)
        
        # Update task status to partially_completed if task_id is available
        if task_id and 'simulation_tasks' in globals() and task_id in simulation_tasks:
            simulation_tasks[task_id]["status"] = "partially_completed" if month < n_months else "completed"
            simulation_tasks[task_id]["completed_months"] = month
            simulation_tasks[task_id]["total_months"] = n_months
            simulation_tasks[task_id]["progress_percentage"] = (month / n_months) * 100
            print(f"📊 Updated task status: {simulation_tasks[task_id]['status']} ({month}/{n_months} months)")
        
        output_dir = 'output'

        # ✅ Collect simulation result files
        file_keys = [
            "behavior_tracker", "discipline_report", "financial_strategy",
            "goal_status", "karmic_tracker", "simulated_cashflow"
        ]
        simulation_outputs = {}
        try:
            # ✅ Load standard simulation result files
            for key in file_keys:
                filename = f"{user_id}_{key}_simulation.json"
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
            reflection_path = os.path.join("data", "reports", reflection_filename)
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

            # Run the async function
            status_code = asyncio.run(notify_frontend())
            print(f"✅ Notified frontend for Month {month} (Status: {status_code})")
            
        except Exception as e:
            print(f"❌ Failed to notify frontend in Month {month}: {e}")
            
    print("\n🎉 All months simulated successfully!")
    return True
