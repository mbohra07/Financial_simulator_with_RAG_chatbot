import numpy as np
import json
import os

def simulate_month(user_name=None, month_index=None):
    try:
        with open(f"output/simulated_cashflow_simulation_{month_index}.json", "r") as f:
            cashflow_data = json.load(f)
    except:
        cashflow_data = {}
    try:
        with open(f"spending_review_simulation_{month_index}.json", "r") as f:
            spending_output = f.read()
    except:
        spending_output = "Not Available"

    try:
        with open(f"output/goal_tracking_simulation_{month_index}.json", "r") as f:
            goal_output = f.read()
    except:
        goal_output = "Not Available"

    try:
        with open(f"output/financial_strategy_simulation_{month_index}.json", "r") as f:
            plan_output = f.read()
    except:
        plan_output = "Not Available"

    try:
        with open(f"output/coordinator_decision_simulation_{month_index}.json", "r") as f:
            final_action = json.load(f)
    except:
        final_action = {}

    try:
        with open(f"output/discipline_tracker_results_simulation_{month_index}.json", "r") as f:
            discipline_tracker = json.load(f)
    except:
        discipline_tracker = {}

    try:
        with open(f"output/behavior_tracker_simulation_{month_index}.json", "r") as f:
            behaviour_tracker = json.load(f)
    except:
        behaviour_tracker = {}

    try:
        with open(f"output/karmic_tracker_simulation_{month_index}.json", "r") as f:
            karma_tracker = json.load(f)
    except:
        karma_tracker = {}

    try:
        with open(f"output/mentor_advice_simulation_{month_index}.json", "r") as f:
            mentor_advice = json.load(f)
    except:
        mentor_advice = {}

    try:
        with open(f"output/monthly_summary_simulation_{month_index}.json", "r") as f:
            monthly_summary = json.load(f)
    except:
        monthly_summary = {}

    monthly_data = {
                    "month_index": month_index,
                    "cashflow": cashflow_data,
                    "spending": spending_output,
                    "goals": goal_output,
                    "Financial_strategy": plan_output,
                    "coordinator decision": final_action,
                    'discipline tracker': discipline_tracker,
                    "behaviour tracker": behaviour_tracker,
                    "karma tracker": karma_tracker,
                    "mentor advice": mentor_advice,
                    "monthly_summary": monthly_summary
                }

                
    timeline_path = f"data/timeline_{user_name}.json"
    os.makedirs(os.path.dirname(timeline_path), exist_ok=True)
    if os.path.exists(timeline_path):
        with open(timeline_path, "r") as f:
            timeline = json.load(f)
    else:
        timeline = []

    timeline.append(monthly_data)

    #if len(timeline) > 3:
        #timeline = timeline[-3:]

    with open(timeline_path, "w") as f:
        json.dump(timeline, f, indent=2)

    return monthly_data

def summary_for_the_month(month_index):
    try:
        with open("output/report.json", "r") as f:
            summary = f.read()
    except:
        summary = "Not Available"
    try:
        with open("output/mentor_task.md", "r") as f:
            mentor_feedback = f.read()
    except:
        mentor_feedback = "Not Available"

    monthly_summary = {
                    "month_index": month_index,
                    "Summary": summary,
                    "mentor Feedback": mentor_feedback
                }
    
    summary_path = f"data/summary{month_index}.json"
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            mon_summary = json.load(f)
    else:
            mon_summary = []

    mon_summary.append(monthly_summary)

    with open(summary_path, "w") as f:
        json.dump(mon_summary, f, indent=2)

    return monthly_summary

