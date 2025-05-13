import os
import json
from collections import Counter

def build_simulated_cashflow_context(month_number, user_id, goal_target=None, goal_text=""):
    file_path = f"output/{user_id}_simulated_cashflow_simulation.json"
    try:
        if not os.path.exists(file_path):
            print(f"⚠️ File {file_path} not found. Treating as empty cashflow history.")
            return "No previous cash flow history available for this user."

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Filter entries for previous months
        prev_entries = [entry for entry in data if entry.get("month") is not None and entry["month"] < month_number]
        if not prev_entries:
            return "No previous cash flow history available for this user."

        # Calculate aggregates
        total_income = sum(entry.get("income", {}).get("total", 0) for entry in prev_entries)
        total_expenses = sum(entry.get("expenses", {}).get("total", 0) for entry in prev_entries)
        total_savings = sum(entry.get("savings", 0) for entry in prev_entries)
        n = len(prev_entries)
        avg_savings = total_savings / n if n else 0
        avg_savings_rate = (avg_savings / (total_income / n)) * 100 if total_income and n else 0

        # Goal progress
        progress_percent = (total_savings / goal_target) * 100 if goal_target else 0

        # Monthly breakdown table
        table_lines = []
        for entry in prev_entries:
            m = entry["month"]
            inc = entry.get("income", {}).get("total", 0)
            exp = entry.get("expenses", {}).get("total", 0)
            sav = entry.get("savings", 0)
            table_lines.append(f"  - Month {m}: Income ₹{inc}, Expenses ₹{exp}, Savings ₹{sav}")

        # Simple trend detection
        savings_list = [entry.get("savings", 0) for entry in prev_entries]
        if all(s == 0 for s in savings_list):
            trend_1 = "No savings in any month."
        elif all(s > 0 for s in savings_list):
            trend_1 = "Consistent savings each month."
        else:
            trend_1 = "Savings pattern is inconsistent."

        if all(entry.get("debt_taken", 0) == 0 for entry in prev_entries):
            trend_2 = "No debt taken so far."
        else:
            trend_2 = "Debt has been taken in some months."

        # Compose context string
        context = (
            f"Improve my responses based on past User's Financial History (Months 1 to {month_number-1}):\n\n"
            f"- Total Months: {n}\n"
            f"- Total Savings: ₹{total_savings}\n"
            f"- Average Monthly Savings: ₹{avg_savings:.2f}\n"
            f"- Total Income: ₹{total_income}\n"
            f"- Total Expenses: ₹{total_expenses}\n"
            f"- Average Savings Rate: {avg_savings_rate:.1f}%\n"
            f"- Goal: {goal_text}\n"
            f"- Progress: Saved ₹{total_savings} towards goal of ₹{goal_target} ({progress_percent:.1f}% complete)\n\n"
            f"Monthly Breakdown:\n" +
            "\n".join(table_lines) +
            "\n\nTrends/Patterns:\n"
            f"- {trend_1}\n"
            f"- {trend_2}"
        )
        return context

    except Exception as e:
        print(f"❗ Error building simulated cashflow context: {e}")
        return f"Error building cashflow context: {e}"


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

    
def build_discipline_report_context(month_number, user_id):
    file_path = f"output/{user_id}_discipline_report_simulation.json"
    try:
        if not os.path.exists(file_path):
            print(f"⚠️ File {file_path} not found. Treating as empty discipline report history.")
            return "No previous discipline report history available for this user."

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        summary_lines = []
        discipline_scores = []
        all_violations = []
        all_recommendations = []

        for entry in data:
            month = entry.get("month")
            if month is not None and month < month_number:
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
                    f"Month {month}: Discipline Score {discipline_score if discipline_score is not None else 'N/A'}. "
                    f"Violations: {violation_str}. Recommendations: {rec_str}"
                )
                summary_lines.append(summary)

        if not summary_lines:
            return "No previous discipline report history available for this user."

        # Aggregate statistics
        n = len(discipline_scores)
        avg_score = sum(discipline_scores) / n if n else 0
        most_common_violations = Counter(all_violations).most_common(2)
        most_common_recs = Counter(all_recommendations).most_common(2)

        context = f"Improve my responses based on past User's Discipline History (Months 1 to {month_number-1}):\n\n"
        context += f"- Total Months: {n}\n"
        context += f"- Average Discipline Score: {avg_score:.2f}\n"
        if most_common_violations:
            context += "- Most Common Violations: " + ", ".join([f"{v[0]} ({v[1]}x)" for v in most_common_violations]) + "\n"
        if most_common_recs:
            context += "- Most Common Recommendations: " + ", ".join([f"{r[0]} ({r[1]}x)" for r in most_common_recs]) + "\n"
        context += "\nMonthly Breakdown:\n" + "\n".join(summary_lines)
        context += "\n\nSummary:\n"
        context += f"- The user's discipline score {'has been consistent' if len(set(discipline_scores)) <= 1 else 'has varied'} over the months."
        context += " Addressing the most frequent violations and following recommendations may help improve discipline."

        return context

    except Exception as e:
        print(f"❗ Error building discipline report context: {e}")
        return f"Error building discipline report context: {e}"
    
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
    
def build_goal_status_context(month_number, user_id):
    file_path = f"output/{user_id}_goal_status_simulation.json"
    try:
        if not os.path.exists(file_path):
            print(f"⚠️ File {file_path} not found. Treating as empty goal status history.")
            return "No previous goal status history available for this user."

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        summary_lines = []
        goal_status_counter = Counter()
        all_adjustments = []
        total_saved = 0
        total_expected = 0
        months_counted = 0

        for entry in data:
            # Handle format {"month": 3, "goals": [{...}], "summary": {...}}
            if isinstance(entry, dict) and "month" in entry and "goals" in entry:
                month = entry.get("month")
                if month is not None and month < month_number:
                    months_counted += 1
                    goal_details = entry.get("goals", [])
                    for goal in goal_details:
                        name = goal.get("name", "N/A")
                        status = goal.get("status", "N/A")
                        saved_so_far = goal.get("saved_so_far", 0)
                        expected_by_now = goal.get("expected_by_now", 0)
                        adjustment_suggestion = goal.get("adjustment_suggestion", "N/A")

                        total_saved += float(saved_so_far) if isinstance(saved_so_far, (int, float)) else 0
                        total_expected += float(expected_by_now) if isinstance(expected_by_now, (int, float)) else 0
                        goal_status_counter[status] += 1
                        all_adjustments.append(adjustment_suggestion)

                        goal_summary = (
                            f"Month {month}: Goal '{name}', Status: {status}, "
                            f"Saved: {saved_so_far}, Expected: {expected_by_now}. "
                            f"Suggestion: {adjustment_suggestion}"
                        )
                        summary_lines.append(goal_summary)

        if not summary_lines:
            return "No previous goal status history available for this user."

        # Most common adjustment suggestions
        common_adjustments = [adj for adj, _ in Counter(all_adjustments).most_common(2) if adj != "N/A"]

        # Compose context
        context = f"Improve my responses based on past User's Goal Status History (Months 1 to {month_number-1}):\n\n"
        context += f"- Months Tracked: {months_counted}\n"
        context += f"- Total Saved Across Goals: ₹{total_saved:.2f}\n"
        context += f"- Total Expected By Now: ₹{total_expected:.2f}\n"
        for status, count in goal_status_counter.items():
            context += f"- Goals with status '{status}': {count}\n"
        if common_adjustments:
            context += f"- Most Common Suggestions: {', '.join(common_adjustments)}\n"
        context += "\nMonthly Breakdown:\n" + "\n".join(summary_lines)
        context += "\n\nSummary:\n"
        if goal_status_counter.get('on track', 0) > goal_status_counter.get('behind', 0):
            context += "- Most goals are on track. Continue your current savings strategy."
        else:
            context += "- Several goals are behind. Consider following the most common suggestions to improve progress."

        return context

    except Exception as e:
        print(f"❗ Error building goal status context: {e}")
        return f"Error building goal status context: {e}"
    
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