import os
import json

def deduplicate_and_save(path, parsed_result):
        try:
            # Load existing data
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            else:
                existing_data = []

            # Extract all existing months
            existing_months = {entry.get("month") for entry in existing_data if isinstance(entry, dict)}

            # Determine new data to append
            new_entries = []
            if isinstance(parsed_result, list):
                for item in parsed_result:
                    if isinstance(item, dict) and item.get("month") not in existing_months:
                        new_entries.append(item)
            elif isinstance(parsed_result, dict):
                if parsed_result.get("month") not in existing_months:
                    new_entries.append(parsed_result)

            if not new_entries:
                print(f"⚠️ No new data to append to '{path}' (duplicate month detected).")
                return

            # Append and write
            existing_data.extend(new_entries)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, indent=4, ensure_ascii=False)
            print(f"💾 Appended result to {path}")
        except Exception as e:
            print(f"❗ Error writing to '{path}': {e}")

def load_json(filepath):
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(filepath, data):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def compute_persona_title(karma_score, behavior_pattern):
    if karma_score > 75 and behavior_pattern == "Consistent Saver":
        return "Wise Sage"
    elif 50 <= karma_score <= 75 and behavior_pattern in ["Consistent Saver", "Occasional Spender"]:
        return "Disciplined Hustler"
    elif karma_score < 50 or behavior_pattern == "Inconsistent Behavior":
        return "Reckless Drifter"
    else:
        return "Balanced Explorer"

def get_month_entries(data, user_name, month):
    # Filter entries for user and month
    return [entry for entry in data if entry.get("user_name") == user_name and entry.get("month") == month]

def generate_monthly_reflection_report(user_name, month):
    # Paths
    output_dir = "output"
    monthly_ouput_dir = "monthly_output"
    data_dir = "data/"
    os.makedirs(data_dir, exist_ok=True)

    # Load logs
    karma_log = load_json(os.path.join(output_dir, "karmic_tracker_simulation.json")) or []
    behavior_log = load_json(os.path.join(output_dir, "behavior_tracker_simulation.json")) or []
    persona_log = load_json(os.path.join("data", "persona_history.json")) or []

    # Filter for current user and month
    karma_entries = get_month_entries(karma_log, user_name, month)
    behavior_entries = get_month_entries(behavior_log, user_name, month)
    persona_entries = get_month_entries(persona_log, user_name, month)

    # Monthly karma score (average)
    if karma_entries:
        karma_scores = [entry.get("traits", {}).get("karma_score", 0) for entry in karma_entries]
        monthly_karma_score = round(sum(karma_scores) / len(karma_scores), 2)
    else:
        monthly_karma_score = None

    # Persona title + transition
    persona_title = persona_entries[-1]["persona_title"] if persona_entries else "Unassigned"
    transition_note = persona_entries[-1].get("transition_reason", "No transition noted") if persona_entries else "No transition noted"

    # Key behavior observations
    if behavior_entries:
        behavior_traits = behavior_entries[-1].get("traits", {})
        behavior_obs = {
            "spending_pattern": behavior_traits.get("spending_pattern", ""),
            "goal_adherence": behavior_traits.get("goal_adherence", ""),
            "saving_consistency": behavior_traits.get("saving_consistency", ""),
            "labels": behavior_traits.get("labels", [])
        }
    else:
        behavior_obs = {}

    # Summary message
    summary_message = f"You’ve evolved into a {persona_title} due to {transition_note}."

    # Assemble report
    report = {
        "month": month,
        "user_name": user_name,
        "monthly_karma_score": monthly_karma_score,
        "persona_title": persona_title,
        "transition_note": transition_note,
        "key_behavior_observations": behavior_obs,
        "summary_message": summary_message
    }

    # Save report with user_id prefix

    # Save to the combined reflection file
    save_monthly_path = os.path.join(data_dir, "reflection_month.json")

    # Check if file exists
    if os.path.exists(save_monthly_path):
        with open(save_monthly_path, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    # Ensure it's a list to append safely
    if not isinstance(existing_data, list):
        existing_data = [existing_data]

    # Append new report
    existing_data.append(report)

    # Write updated list back to file
    with open(save_monthly_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)

    # Create monthly_output directory if it doesn't exist
    os.makedirs(monthly_ouput_dir, exist_ok=True)

    # Save individual reflection file with user_id prefix only
    save_path = os.path.join(monthly_ouput_dir, f"{user_name}_reflection_month_{month}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"💾 Monthly reflection saved with user_id prefix: {save_path}")
    print(f"✅ Reflection report saved: {save_path}")
    return report

def assign_persona(user_name, month):
    # Try to load karma data with user prefix
    karma_file = f'output/{user_name}_karmic_tracker_simulation.json'
    if os.path.exists(karma_file):
        karma_data = load_json(karma_file)
    else:
        # Fallback to non-prefixed file
        fallback_karma_file = 'output/karmic_tracker_simulation.json'
        if os.path.exists(fallback_karma_file):
            karma_data = load_json(fallback_karma_file)
        else:
            karma_data = []

    # Try to load behavior data with user prefix
    behavior_file = f'output/{user_name}_behavior_tracker_simulation.json'
    if os.path.exists(behavior_file):
        behavior_data = load_json(behavior_file)
    else:
        # Fallback to non-prefixed file
        fallback_behavior_file = 'output/behavior_tracker_simulation.json'
        if os.path.exists(fallback_behavior_file):
            behavior_data = load_json(fallback_behavior_file)
        else:
            behavior_data = []

    # Try to load user-specific person_history file first
    user_history_path = f'data/{user_name}_person_history.json'
    if os.path.exists(user_history_path):
        history_data = load_json(user_history_path)
    else:
        # If not found, create a new empty list
        history_data = []

    # Extract average karma score for the month
    karmic_scores = [entry.get('traits', {}).get('karma_score', 0) for entry in karma_data if entry.get('user_name') == user_name and entry.get('month') == month]
    avg_karmic_score = sum(karmic_scores) / len(karmic_scores) if karmic_scores else 50

    # Extract behavior pattern
    behavior_entry = next((entry for entry in behavior_data if entry.get('user_name') == user_name and entry.get('month') == month), None)
    behavior_pattern = behavior_entry.get('traits', {}).get('spending_pattern', "Inconsistent Behavior") if behavior_entry else "Inconsistent Behavior"

    persona_title = compute_persona_title(avg_karmic_score, behavior_pattern)

    # Check if persona changed
    last_persona = history_data[-1]['persona_title'] if history_data else None
    change_flag = persona_title != last_persona

    record = {
        "month": month,
        "user_name": user_name,
        "persona_title": persona_title,
        "avg_karmic_score": avg_karmic_score,
        "behavior_pattern": behavior_pattern,
        "change_flag": change_flag
    }

    # Check if an entry for this month already exists
    existing_entry_index = next((i for i, entry in enumerate(history_data) if entry.get('month') == month), None)

    if existing_entry_index is not None:
        # Update existing entry
        history_data[existing_entry_index] = record
        print(f"🔄 Updated existing persona history entry for month {month}")
    else:
        # Append new entry
        history_data.append(record)
        print(f"➕ Added new persona history entry for month {month}")

    # Save with user_id prefix only - use person_history for consistency with API
    persona_history_path = f'data/{user_name}_person_history.json'
    save_json(persona_history_path, history_data)
    print(f"💾 Persona history saved with user_id prefix: {persona_history_path}")

    print(f"🔮 Persona Assigned for Month {month}: {persona_title} (Change: {change_flag})")
    return record