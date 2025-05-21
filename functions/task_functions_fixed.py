import os
import json
from collections import Counter

def build_simulated_cashflow_context(month_number, user_id, flag=True):
    file_path = f"output/{user_id}_simulated_cashflow_simulation.json"
    try:
        if not os.path.exists(file_path):
            print(f"⚠️ File {file_path} not found. Treating as empty cashflow history.")
            return "No previous cash flow history available for this user."

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"✅ Loaded data for user {user_id}. month_number={month_number}, flag={flag}")

        # Ensure month_number is int
        if isinstance(month_number, str) and month_number.isdigit():
            month_number = int(month_number)

        # Ensure data is a list
        if not isinstance(data, list):
            print(f"⚠️ Expected list but got {type(data)}")
            return "Invalid data format in cashflow history file."

        # Filter previous or previous + current month entries
        prev_entries = []
        for entry in data:
            m = entry.get("month")
            if isinstance(m, str) and m.isdigit():
                m = int(m)
            if isinstance(m, int) and ((m < month_number) or (flag and m == month_number)):
                entry["month"] = m
                prev_entries.append(entry)

        if not prev_entries:
            return "No relevant cash flow history available for this user."

        print(f"📦 Found {len(prev_entries)} entries for user_id={user_id}")

        # Aggregates
        total_income = sum(entry.get("income", {}).get("total", 0) for entry in prev_entries)
        total_expenses = sum(entry.get("expenses", {}).get("total", 0) for entry in prev_entries)

        # Handle different savings formats
        total_savings = 0
        for entry in prev_entries:
            savings = entry.get("savings", 0)
            if isinstance(savings, dict):
                total_savings += savings.get("amount", 0)
            elif isinstance(savings, (int, float)):
                total_savings += savings

        n = len(prev_entries)
        avg_savings = total_savings / n if n else 0
        avg_income = total_income / n if n else 0
        avg_savings_rate = (avg_savings / avg_income) * 100 if avg_income else 0

        # Monthly Table
        table_lines = []
        for entry in prev_entries:
            m = entry["month"]
            inc = entry.get("income", {}).get("total", 0)
            exp = entry.get("expenses", {}).get("total", 0)

            # Handle different savings formats
            savings = entry.get("savings", 0)
            if isinstance(savings, dict):
                sav = savings.get("amount", 0)
            elif isinstance(savings, (int, float)):
                sav = savings
            else:
                sav = 0

            table_lines.append(f"  - Month {m}: Income ₹{inc}, Expenses ₹{exp}, Savings ₹{sav}")

        # Trend Analysis
        savings_list = []
        for entry in prev_entries:
            savings = entry.get("savings", 0)
            if isinstance(savings, dict):
                savings_list.append(savings.get("amount", 0))
            elif isinstance(savings, (int, float)):
                savings_list.append(savings)
            else:
                savings_list.append(0)

        debt_list = [entry.get("debt_taken", 0) for entry in prev_entries]

        if all(s == 0 for s in savings_list):
            trend_1 = "User did not save any money in any of the months."
        elif all(s > 0 for s in savings_list):
            trend_1 = "User consistently saved every month."
        else:
            trend_1 = "User's saving habits were inconsistent."

        if all(d == 0 for d in debt_list):
            trend_2 = "User has not taken on any debt."
        else:
            trend_2 = "User has taken on debt in some months."

        # Final Context Prompt
        context = (
            f"You are an intelligent financial assistant helping a user build better financial habits.\n"
            f"Use their actual financial journey below to deliver more personalized, empathetic, and context-aware feedback.\n\n"

            f"📊 Summary of User's Financial Journey (Month 1 to {month_number}):\n"
            f"- Total Months Tracked: {n}\n"
            f"- Total Income: ₹{total_income}\n"
            f"- Total Expenses: ₹{total_expenses}\n"
            f"- Total Savings: ₹{total_savings}\n"
            f"- Average Monthly Savings: ₹{avg_savings:.2f}\n"
            f"- Average Savings Rate: {avg_savings_rate:.1f}%\n\n"

            f"📅 Monthly Financial Overview:\n"
            f"{chr(10).join(table_lines)}\n\n"

            f"📈 Detected Behavioral Trends:\n"
            f"1. {trend_1}\n"
            f"2. {trend_2}\n\n"

            f"🧠 INSTRUCTION:\n"
            f"Analyze the user's behavior based on the patterns above. Your response should:\n"
            f"- Reflect historical progress or pitfalls.\n"
            f"- Recognize consistency and reward improvements.\n"
            f"- Offer constructive advice where habits need correction.\n"
            f"- Reference specific months if relevant.\n"
            f"- Be encouraging, human, and actionable.\n\n"

            f"💡 Now, using all this context, generate a response that sounds like a smart, empathetic mentor who is genuinely trying to help the user make better financial decisions."
        )
        return context

    except json.JSONDecodeError as je:
        print(f"❗ JSON decode error: {je}")
        return "Error parsing cashflow data: Invalid JSON format."
    except Exception as e:
        print(f"❗ Error building simulated cashflow context: {e}")
        return f"Error building cashflow context: {e}"
