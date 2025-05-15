import os
import json
import time


def save_to_json_file(file_path, new_data):
    """Save data to a JSON file, appending to existing data if file exists."""
    try:
        # Load existing data if file exists
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        else:
            existing_data = []

        # Append or extend based on data type
        if isinstance(new_data, list):
            existing_data.extend(new_data)
        else:
            existing_data.append(new_data)

        # Write updated data back to file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=4, ensure_ascii=False)

        print(f"💾 Saved result to {file_path}")
    except Exception as e:
        print(f"❗ Error saving to file '{file_path}': {e}")


def save_task_outputs(task, result, user_id, month_number):
    """Save task results to output files."""
    output_dir = os.path.dirname(task.output_file)
    monthly_output_dir = "monthly_output"
    base_file_name = os.path.basename(task.output_file)
    file_name_without_ext, ext = os.path.splitext(base_file_name)
    
    # Create file paths
    monthly_file_name = f"{user_id}_{file_name_without_ext}_simulation_{month_number}{ext}"
    dynamic_file_name = f"{user_id}_{file_name_without_ext}_simulation{ext}"
    monthly_output_path = os.path.join(monthly_output_dir, monthly_file_name)
    output_path = os.path.join(output_dir, dynamic_file_name)

    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(monthly_output_dir, exist_ok=True)

    # Save to main output file
    save_to_json_file(output_path, result)
    
    # Save to monthly output file
    save_to_json_file(monthly_output_path, result)

def execute_task_with_retries(task, inputs, max_retries):
    """Execute a task with retries for JSON parsing."""
    retries = 0
    task_result = None
    parsed_result = None
    
    while retries < max_retries:
        task_result = task.agent.execute_task(task, inputs)
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
        
    return parsed_result

def check_tool_status(task):
    """Check if a task's tool is working properly."""
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

