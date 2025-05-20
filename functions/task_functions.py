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
        total_savings = sum(entry.get("savings", 0) for entry in prev_entries)
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
            sav = entry.get("savings", 0)
            table_lines.append(f"  - Month {m}: Income ₹{inc}, Expenses ₹{exp}, Savings ₹{sav}")

        # Trend Analysis
        savings_list = [entry.get("savings", 0) for entry in prev_entries]
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
        context = f"""You are an intelligent financial assistant helping a user build better financial habits.\n
        Use their actual financial journey below to deliver more personalized, empathetic, and context-aware feedback.\n\n📊
        Summary of User’s Financial Journey (Month 1 to {month_number}):\n
        - Total Months Tracked: {n}\n
        - Total Income: ₹{total_income}\n
        - Total Expenses: ₹{total_expenses}\n
        - Total Savings: ₹{total_savings}\n
        - Average Monthly Savings: ₹{avg_savings:.2f}\n
        - Average Savings Rate: {avg_savings_rate:.1f}%\n\n
        📅 Monthly Financial Overview:\n" +
        "\n"{table_lines} +
        "\n\n📈 Detected Behavioral Trends:\n1.
        {trend_1}\n2.
        {trend_2}\n\n🧠
        INSTRUCTION:\nAnalyze the user's behavior based on the patterns above.
        Your response should:\nR
        eflect historical progress or pitfalls.\n
        Recognize consistency and reward improvements.\n
        Offer constructive advice where habits need correction.\n
        Reference specific months if relevant.\nBe encouraging, human, and actionable.\n\n
        💡 Now, using all this context, generate a response that sounds like a smart, empathetic mentor who is genuinely trying to help the user make better financial decisions."""
        return context

    except json.JSONDecodeError as je:
        print(f"❗ JSON decode error: {je}")
        return "Error parsing cashflow data: Invalid JSON format."
    except Exception as e:
        print(f"❗ Error building simulated cashflow context: {e}")
        return f"Error building cashflow context: {e}"

def build_discipline_report_context(month_number, user_id, flag=True):
    file_path = f"output/{user_id}_discipline_report_simulation.json"
    try:
        if not os.path.exists(file_path):
            print(f"⚠️ File {file_path} not found. Treating as empty discipline report history.")
            return "No previous discipline report history available for this user."

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"✅ Loaded discipline data for user {user_id}. month_number={month_number}, flag={flag}")

        # Ensure month_number is int
        if isinstance(month_number, str) and month_number.isdigit():
            month_number = int(month_number)

        # Ensure data is a list
        if not isinstance(data, list):
            print(f"⚠️ Expected list but got {type(data)}")
            return "Invalid data format in discipline report file."

        # Filter previous or previous + current month entries
        relevant_entries = []
        for entry in data:
            m = entry.get("month")
            if isinstance(m, str) and m.isdigit():
                m = int(m)
            if isinstance(m, int) and ((m < month_number) or (flag and m == month_number)):
                entry["month"] = m
                relevant_entries.append(entry)

        if not relevant_entries:
            return "No relevant discipline report history available for this user."

        print(f"📦 Found {len(relevant_entries)} discipline entries for user_id={user_id}")

        # Process entries
        summary_lines = []
        discipline_scores = []
        all_violations = []
        all_recommendations = []

        for entry in relevant_entries:
            month = entry.get("month")
            violations = entry.get("violations", [])
            discipline_score = entry.get("discipline_score", None)
            recommendations = entry.get("recommendations", [])

            if discipline_score is not None:
                discipline_scores.append(discipline_score)
            all_violations.extend(violations)
            all_recommendations.extend(recommendations)

            violation_str = "; ".join(violations) if violations else "No violations"
            rec_str = "; ".join(recommendations) if recommendations else "No recommendations"

            summary = (
                f"  - Month {month}: Score {discipline_score if discipline_score is not None else 'N/A'}. "
                f"Violations: {violation_str}. Recommendations: {rec_str}"
            )
            summary_lines.append(summary)

        # Aggregate statistics
        n = len(discipline_scores)
        avg_score = sum(discipline_scores) / n if n else 0
        most_common_violations = Counter(all_violations).most_common(3)
        most_common_recs = Counter(all_recommendations).most_common(3)

        # Trend analysis
        if n > 1:
            score_trend = "improving" if discipline_scores[-1] > discipline_scores[0] else "declining" if discipline_scores[-1] < discipline_scores[0] else "stable"
            consistency = "consistent" if max(discipline_scores) - min(discipline_scores) <= 1 else "inconsistent"
        else:
            score_trend = "not established yet"
            consistency = "not established yet"

        # Final Context Prompt
        context = (
            f"You are an intelligent financial discipline coach helping a user build better financial habits.\n"
            f"Use their actual discipline journey below to deliver more personalized, empathetic, and context-aware feedback.\n\n"

            f"📊 Summary of User's Financial Discipline (Month 1 to {month_number}):\n"
            f"- Total Months Tracked: {n}\n"
            f"- Average Discipline Score: {avg_score:.2f}/10\n"
            f"- Discipline Trend: {score_trend}\n"
            f"- Consistency: {consistency}\n"
        )

        if most_common_violations:
            context += "- Most Common Violations:\n"
            for v, count in most_common_violations:
                context += f"  • {v} ({count}x)\n"

        if most_common_recs:
            context += "- Most Common Recommendations:\n"
            for r, count in most_common_recs:
                context += f"  • {r} ({count}x)\n"

        context += (
            f"\n📅 Monthly Discipline Breakdown:\n" +
            "\n".join(summary_lines) + "\n\n"

            f"📈 Behavioral Analysis:\n"
            f"1. The user's discipline score is {score_trend} over time.\n"
            f"2. The user has been {consistency} in following financial rules.\n"
            f"3. The most frequent issue is {most_common_violations[0][0] if most_common_violations else 'not yet established'}.\n\n"

            f"🧠 INSTRUCTION:\n"
            f"Analyze the user's financial discipline based on the patterns above. Your response should:\n"
            f"- Acknowledge their discipline journey and progress or challenges.\n"
            f"- Provide specific advice addressing their most common violations.\n"
            f"- Reinforce previous recommendations that haven't been addressed.\n"
            f"- Be encouraging but firm about the importance of financial discipline.\n"
            f"- Reference specific months if relevant to show patterns.\n\n"

            f"💡 Now, using all this context, generate a response that sounds like a supportive but firm financial coach who wants to help the user improve their financial discipline."
        )

        return context

    except json.JSONDecodeError as je:
        print(f"❗ JSON decode error: {je}")
        return "Error parsing discipline data: Invalid JSON format."
    except Exception as e:
        print(f"❗ Error building discipline report context: {e}")
        return f"Error building discipline report context: {e}"

def build_goal_status_context(month_number, user_id, flag=True):
    file_path = f"output/{user_id}_goal_status_simulation.json"
    try:
        if not os.path.exists(file_path):
            print(f"⚠️ File {file_path} not found. Treating as empty goal status history.")
            return "No previous goal status history available for this user."

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"✅ Loaded goal status data for user {user_id}. month_number={month_number}, flag={flag}")

        # Ensure month_number is int
        if isinstance(month_number, str) and month_number.isdigit():
            month_number = int(month_number)

        # Ensure data is a list
        if not isinstance(data, list):
            print(f"⚠️ Expected list but got {type(data)}")
            return "Invalid data format in goal status file."

        # Filter previous or previous + current month entries
        relevant_entries = []
        for entry in data:
            if not isinstance(entry, dict) or "month" not in entry or "goals" not in entry:
                continue

            m = entry.get("month")
            if isinstance(m, str) and m.isdigit():
                m = int(m)
            if isinstance(m, int) and ((m < month_number) or (flag and m == month_number)):
                entry["month"] = m
                relevant_entries.append(entry)

        if not relevant_entries:
            return "No relevant goal status history available for this user."

        print(f"📦 Found {len(relevant_entries)} goal status entries for user_id={user_id}")

        # Process entries
        summary_lines = []
        goal_status_counter = Counter()
        all_adjustments = []
        total_saved = 0
        total_expected = 0
        goal_progress = {}  # Track progress of each goal over time

        for entry in relevant_entries:
            month = entry.get("month")
            goals_data = entry.get("goals", [])

            # Handle both formats: list of goals or dictionary of goals
            if isinstance(goals_data, dict):
                # Convert dictionary format to list format for processing
                goals_list = []
                for goal_name, goal_info in goals_data.items():
                    # Create a goal object in the expected format
                    goal_obj = {
                        "name": goal_name,
                        "status": goal_info.get("status", "N/A"),
                        "saved_so_far": goal_info.get("progress", 0),  # Use progress as saved_so_far
                        "expected_by_now": goal_info.get("target", 0) * 0.8,  # Estimate expected as 80% of target
                        "target_amount": goal_info.get("target", 0),
                        "adjustment_suggestion": goal_info.get("adjustment", "N/A")
                    }
                    goals_list.append(goal_obj)
                goal_details = goals_list
            else:
                # Already in list format
                goal_details = goals_data

            for goal in goal_details:
                # Skip if not a dictionary
                if not isinstance(goal, dict):
                    continue

                name = goal.get("name", "N/A")
                status = goal.get("status", "N/A")
                saved_so_far = goal.get("saved_so_far", 0)
                expected_by_now = goal.get("expected_by_now", 0)
                target_amount = goal.get("target_amount", 0)
                adjustment_suggestion = goal.get("adjustment_suggestion", goal.get("adjustment", "N/A"))

                # Convert to numeric if needed
                saved_so_far = float(saved_so_far) if isinstance(saved_so_far, (int, float, str)) and str(saved_so_far).replace('.', '', 1).isdigit() else 0
                expected_by_now = float(expected_by_now) if isinstance(expected_by_now, (int, float, str)) and str(expected_by_now).replace('.', '', 1).isdigit() else 0
                target_amount = float(target_amount) if isinstance(target_amount, (int, float, str)) and str(target_amount).replace('.', '', 1).isdigit() else 0

                # Update totals
                total_saved += saved_so_far
                total_expected += expected_by_now
                goal_status_counter[status] += 1

                if adjustment_suggestion and adjustment_suggestion != "N/A":
                    all_adjustments.append(adjustment_suggestion)

                # Track goal progress over time
                if name not in goal_progress:
                    goal_progress[name] = []
                goal_progress[name].append({
                    "month": month,
                    "saved": saved_so_far,
                    "expected": expected_by_now,
                    "target": target_amount,
                    "status": status
                })

                # Format progress percentage
                progress_pct = (saved_so_far / target_amount * 100) if target_amount > 0 else 0
                expected_pct = (expected_by_now / target_amount * 100) if target_amount > 0 else 0

                summary = (
                    f"  - Month {month}: Goal '{name}', Status: {status}, "
                    f"Saved: ₹{saved_so_far:.2f} ({progress_pct:.1f}%), "
                    f"Expected: ₹{expected_by_now:.2f} ({expected_pct:.1f}%). "
                    f"Suggestion: {adjustment_suggestion}"
                )
                summary_lines.append(summary)

        # Analyze goal trends
        goal_trends = []
        for name, progress in goal_progress.items():
            if len(progress) > 1:
                first_status = progress[0]["status"]
                last_status = progress[-1]["status"]

                # Calculate trend
                if first_status == last_status:
                    trend = f"Goal '{name}' has remained {last_status}"
                elif (first_status == "behind" and last_status == "on track") or (first_status == "on track" and last_status == "ahead"):
                    trend = f"Goal '{name}' has improved from {first_status} to {last_status}"
                else:
                    trend = f"Goal '{name}' has declined from {first_status} to {last_status}"

                # Calculate savings rate
                if len(progress) > 1:
                    first_saved = progress[0]["saved"]
                    last_saved = progress[-1]["saved"]
                    months_diff = progress[-1]["month"] - progress[0]["month"]
                    if months_diff > 0:
                        monthly_save_rate = (last_saved - first_saved) / months_diff
                        trend += f", saving ₹{monthly_save_rate:.2f}/month"

                goal_trends.append(trend)

        # Most common adjustment suggestions
        common_adjustments = [adj for adj, count in Counter(all_adjustments).most_common(3) if adj != "N/A"]

        # Calculate overall progress metrics
        progress_ratio = (total_saved / total_expected * 100) if total_expected > 0 else 0
        progress_status = "ahead of schedule" if progress_ratio > 105 else "on track" if progress_ratio >= 95 else "slightly behind" if progress_ratio >= 80 else "significantly behind"

        # Final Context Prompt
        context = (
            f"You are an intelligent financial goals coach helping a user achieve their financial objectives.\n"
            f"Use their actual goal progress below to deliver more personalized, empathetic, and context-aware feedback.\n\n"

            f"📊 Summary of User's Financial Goals (Month 1 to {month_number}):\n"
            f"- Months Tracked: {len(relevant_entries)}\n"
            f"- Total Saved Across All Goals: ₹{total_saved:.2f}\n"
            f"- Total Expected By Now: ₹{total_expected:.2f}\n"
            f"- Overall Progress: {progress_ratio:.1f}% ({progress_status})\n"
        )

        # Add status counts
        for status, count in goal_status_counter.items():
            context += f"- Goals with status '{status}': {count}\n"

        # Add common adjustments
        if common_adjustments:
            context += "- Most Common Suggestions:\n"
            for adj in common_adjustments:
                context += f"  • {adj}\n"

        # Add goal trends
        if goal_trends:
            context += "\n📈 Goal Trends:\n"
            for trend in goal_trends:
                context += f"- {trend}\n"

        # Add monthly breakdown
        context += (
            f"\n📅 Monthly Goal Breakdown:\n" +
            "\n".join(summary_lines) + "\n\n"

            f"📈 Behavioral Analysis:\n"
            f"1. The user is {progress_status} on their overall financial goals.\n"
            f"2. The most common goal status is '{goal_status_counter.most_common(1)[0][0]}' ({goal_status_counter.most_common(1)[0][1]} goals).\n"
            f"3. {goal_trends[0] if goal_trends else 'Goal trend analysis not available yet.'}.\n\n"

            f"🧠 INSTRUCTION:\n"
            f"Analyze the user's financial goals based on the patterns above. Your response should:\n"
            f"- Acknowledge their progress journey and celebrate wins or address challenges.\n"
            f"- Provide specific advice addressing their most common adjustment suggestions.\n"
            f"- Offer strategies to improve goals that are behind schedule.\n"
            f"- Be encouraging about goals that are on track or ahead.\n"
            f"- Reference specific goals by name to personalize your advice.\n\n"

            f"💡 Now, using all this context, generate a response that sounds like a supportive financial goals coach who wants to help the user achieve their financial objectives."
        )

        return context

    except json.JSONDecodeError as je:
        print(f"❗ JSON decode error: {je}")
        return "Error parsing goal status data: Invalid JSON format."
    except Exception as e:
        print(f"❗ Error building goal status context: {e}")
        return f"Error building goal status context: {e}"

def build_financial_strategy_context(month_number, user_id):
    file_path = f"output/{user_id}_financial_strategy_simulation.json"
    try:
        if not os.path.exists(file_path):
            print(f"⚠️ File {file_path} not found. Treating as empty strategy history.")
            return "No previous financial strategy history available for this user."

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        summary_lines = []
        all_recommendations = []

        for entry in data:
            month = entry.get("month")
            if month is not None and month < month_number:
                # Handle both possible structures
                if "traits" in entry:
                    recommendations = entry["traits"].get("recommendations", [])
                    reasoning = entry["traits"].get("reasoning", "")
                else:
                    recommendations = entry.get("recommendations", [])
                    reasoning = entry.get("reasoning", "")

                all_recommendations.extend(recommendations)
                rec_str = "; ".join(recommendations) if recommendations else "No recommendations provided"

                summary = (
                    f"Month {month}: Recommendations: {rec_str}. Reasoning: {reasoning}"
                )
                summary_lines.append(summary)

        if not summary_lines:
            return "No previous financial strategy history available for this user."

        context = f"Improve my responses based on past User's Financial Strategy History (Months 1 to {month_number-1}):\n\n"
        context += "\n".join(summary_lines)

        if all_recommendations:
            unique_recs = set(all_recommendations)
            context += "\n\nCommon Recommendations:\n"
            for rec in unique_recs:
                context += f"- {rec}\n"

        context += "\nSummary:\n"
        context += f"- Reviewed {len(summary_lines)} months of financial strategy data.\n"
        context += "- Recommendations are based on user's financial goals and risk profile."

        return context

    except Exception as e:
        print(f"⛔ Error building financial strategy context: {e}")
        return f"Error building financial strategy context: {e}"

def build_karmic_tracker_context(month_number, user_id):
    file_path = f"output/{user_id}_karmic_tracker_simulation.json"
    try:
        if not os.path.exists(file_path):
            print(f"⚠️ File {file_path} not found. Treating as empty karmic tracker history.")
            return "No previous karmic tracker history available for this user."

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        summary_lines = []
        karma_scores = []
        sattvic_traits = []
        rajasic_traits = []
        tamasic_traits = []
        trends = []

        for entry in data:
            # Handle format {"month": 3, "traits": {...}}
            if isinstance(entry, dict) and "month" in entry and "traits" in entry:
                month = entry.get("month")
                if month is not None and month < month_number:
                    traits = entry.get("traits", {})
                    sattvic = traits.get("sattvic_traits", [])
                    rajasic = traits.get("rajasic_traits", [])
                    tamasic = traits.get("tamasic_traits", [])
                    karma_score = traits.get("karma_score", None)
                    trend = traits.get("trend", "Unknown")

                    if karma_score is not None:
                        karma_scores.append(karma_score)
                    sattvic_traits.extend(sattvic)
                    rajasic_traits.extend(rajasic)
                    tamasic_traits.extend(tamasic)
                    trends.append(trend)

                    summary = (
                        f"Month {month}: Karma Score {karma_score if karma_score is not None else 'N/A'}, Trend {trend}. "
                        f"Sattvic: {', '.join(sattvic) if sattvic else 'None'}. "
                        f"Rajasic: {', '.join(rajasic) if rajasic else 'None'}. "
                        f"Tamasic: {', '.join(tamasic) if tamasic else 'None'}."
                    )
                    summary_lines.append(summary)

        if not summary_lines:
            return "No previous karmic tracker history available for this user."

        # Aggregate stats
        n = len(karma_scores)
        avg_karma = sum(karma_scores) / n if n else 0
        most_common_sattvic = [t for t, _ in Counter(sattvic_traits).most_common(2)]
        most_common_rajasic = [t for t, _ in Counter(rajasic_traits).most_common(2)]
        most_common_tamasic = [t for t, _ in Counter(tamasic_traits).most_common(2)]
        most_common_trend = Counter(trends).most_common(1)[0][0] if trends else "Unknown"

        context = f"Improve my responses based on past User's Karmic History (Months 1 to {month_number-1}):\n\n"
        context += f"- Total Months: {n}\n"
        context += f"- Average Karma Score: {avg_karma:.2f}\n"
        context += f"- Most Common Sattvic Traits: {', '.join(most_common_sattvic) if most_common_sattvic else 'None'}\n"
        context += f"- Most Common Rajasic Traits: {', '.join(most_common_rajasic) if most_common_rajasic else 'None'}\n"
        context += f"- Most Common Tamasic Traits: {', '.join(most_common_tamasic) if most_common_tamasic else 'None'}\n"
        context += f"- Most Common Trend: {most_common_trend}\n"
        context += "\nMonthly Breakdown:\n" + "\n".join(summary_lines)
        context += "\n\nSummary:\n"
        context += "- The user's karmic traits and score reflect behavioral patterns over time. Encourage more sattvic traits for positive financial karma."

        return context

    except Exception as e:
        print(f"❗ Error building karmic tracker context: {e}")
        return f"Error building karmic tracker context: {e}"


def build_behavior_tracker_context(month_number, user_id):
    file_path = f"output/{user_id}_behavior_tracker_simulation.json"
    try:
        if not os.path.exists(file_path):
            print(f"⚠️ File {file_path} not found. Treating as empty behavior tracker history.")
            return "No previous behavior tracker history available for this user."

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        behavior_summary_lines = []
        spending_patterns = []
        goal_adherences = []
        saving_consistencies = []
        all_labels = []

        for entry in data:
            # Handle format {"month": 3, "traits": {...}}
            if isinstance(entry, dict) and "month" in entry and "traits" in entry:
                month = entry.get("month")
                if month is not None and month < month_number:
                    traits = entry.get("traits", {})
                    spending_pattern = traits.get("spending_pattern", "N/A")
                    goal_adherence = traits.get("goal_adherence", "N/A")
                    saving_consistency = traits.get("saving_consistency", "N/A")
                    labels = traits.get("labels", [])
                    labels_str = ", ".join(labels) if labels else "N/A"

                    spending_patterns.append(spending_pattern)
                    goal_adherences.append(goal_adherence)
                    saving_consistencies.append(saving_consistency)
                    all_labels.extend(labels)

                    behavior_summary = (
                        f"Month {month}: Spending Pattern: {spending_pattern}, "
                        f"Goal Adherence: {goal_adherence}, Saving Consistency: {saving_consistency}. "
                        f"Labels: {labels_str}"
                    )
                    behavior_summary_lines.append(behavior_summary)

        if not behavior_summary_lines:
            return "No previous behavior tracker history available for this user."

        # Aggregate trends
        n = len(behavior_summary_lines)
        most_common_spending = Counter(spending_patterns).most_common(1)[0][0] if spending_patterns else "N/A"
        most_common_goal = Counter(goal_adherences).most_common(1)[0][0] if goal_adherences else "N/A"
        most_common_saving = Counter(saving_consistencies).most_common(1)[0][0] if saving_consistencies else "N/A"
        most_common_labels = [label for label, _ in Counter(all_labels).most_common(2)]

        context = f"Improve my responses based on past User's Behavioral Trends (Months 1 to {month_number-1}):\n\n"
        context += f"- Months Tracked: {n}\n"
        context += f"- Most Common Spending Pattern: {most_common_spending}\n"
        context += f"- Most Common Goal Adherence: {most_common_goal}\n"
        context += f"- Most Common Saving Consistency: {most_common_saving}\n"
        if most_common_labels:
            context += f"- Frequent Labels: {', '.join(most_common_labels)}\n"
        context += "\nMonthly Breakdown:\n" + "\n".join(behavior_summary_lines)
        context += "\n\nSummary:\n"
        context += "- The user's behavioral patterns reflect their approach to spending, saving, and goal adherence. Leverage these trends to personalize future recommendations."

        return context

    except Exception as e:
        print(f"❗ Error building behavior tracker context: {e}")
        return f"Error building behavior tracker context: {e}"