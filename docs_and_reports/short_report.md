# Short Report

## Introduction

This project implements an AI-powered personal finance simulator that delivers actionable insights into spending, savings, and financial goals. The system uses **10 specialized agents**, each handling a distinct aspect of financial management, and executes tasks in a sequential, context-aware workflow. The simulator adapts to macroeconomic changes such as inflation, interest rates, and cost of living, and provides a real-time Financial Wellness Score to track user progress. All simulations are run using free API access, with time delays between tasks to avoid rate limit errors.

---

## Agents and Their Logic

- **Spending Advisor:**  
  Acts as an AI budget coach, helping the user identify and reduce non-essential spending using economic reasoning and actionable tips. It leverages concepts like marginal utility, opportunity cost, and habit formation to improve savings.

- **Goal Tracker:**  
  Monitors financial targets and suggests tactical allocation shifts. It provides periodic updates on progress and recommends essential actions to accelerate goal achievement.

- **Financial Strategy Agent:**  
  Analyzes the user’s financial profile and allocates funds optimally across debt repayment, investments, and emergency reserves. It plans holistically and adjusts recommendations based on real-world macroeconomic constraints.

- **Emotional Bias Agent:**  
  Simulates emotional and psychological triggers-such as impulse purchases, guilt spending, or financial anxiety-that influence user behavior. This agent introduces realistic, random spending spikes or savings dips in response to simulated life events.

- **Agent Coordinator:**  
  Synthesizes recommendations from all financial agents and decides on the most prudent, actionable next step for the user. It weighs agent advice, resolves conflicts, and ensures the user always receives a clear, prioritized action.

- **Discipline Tracker Agent:**  
  Evaluates financial discipline by analyzing savings achievements, overspending incidents, and consistency over time. It computes an overall discipline score (0 to 100) and provides actionable tips for improvement.

- **Behavior Tracker Agent:**  
  Monitors, records, and detects patterns in user financial behavior each month. It categorizes the user into distinct behavior patterns, such as "Consistent Saver" or "Occasional Spender," and generates cumulative reports for long-term analysis.

- **Karma Tracker Agent:**  
  Monitors the karmic quality of financial decisions by assigning symbolic value based on sattvic (pure), rajasic (desire-driven), and tamasic (inert/ignorant) traits. It classifies financial actions and generates a rolling "Karmic Score" to encourage balanced and intentional choices.

- **Mentor Agent:**  
  Offers reflective advice on financial habits, focusing on wins, mistakes, and long-term growth strategies. It synthesizes behavioral trends and discipline metrics from other agents to deliver personalized mentorship and spiritual guidance.

- **Monthly Summary Agent:**  
  Consolidates all agent outputs into a comprehensive monthly report, highlighting trends, key insights, and personalized recommendations for improvement.

---

## System Architecture and Sequential Workflow

The simulator is built on the Crew AI framework, allowing modular agent and task management. Each task is executed **sequentially**, with the output of one agent serving as context for the next. This ensures every decision is informed by the latest, most relevant data and maintains a realistic simulation of financial life.

A key feature is the **monthly simulation logic**:  
- The workflow is run in monthly cycles.  
- Each month’s simulation uses the outputs of the previous month’s agents as context, allowing financial state, behavior patterns, and macroeconomic conditions to evolve over time.
- This same sequential logic can be adapted to daily simulations if desired.

For example, after simulating cash flow, the results are passed to the spending evaluation, goal tracking, and financial strategy agents. Their combined outputs are then merged by the agent coordinator, and the subsequent agents (discipline tracker, mentor, karma tracker, behavior tracker) all use this consolidated context. The monthly summary agent aggregates all prior outputs for a holistic view.

The context dependencies for each agent and task are explicitly defined, such as:
- Spending evaluation, goal tracking, and financial strategy depend on the cash flow simulation.
- The coordinator depends on the outputs of these three.
- The discipline tracker, mentor, karma tracker, and behavior tracker depend on all previous outputs.
- The summary agent aggregates the results from all agents.

This design ensures that each agent is context-aware and that the simulation reflects the evolving financial and behavioral state of the user.

---

## Tasks and Simulations

- **Simulate Cash Flow:** Generates inflow/outflow records, including emotional and random life events.
- **Evaluate Spending:** Identifies inefficiencies and suggests improvements.
- **Track Goals:** Monitors progress and projects time to achievement.
- **Financial Strategy:** Recommends optimal fund allocation.
- **Coordinator Decision:** Synthesizes agent outputs into a clear recommendation.
- **Discipline & Behavior Tracking:** Scores and categorizes financial discipline and behavioral patterns.
- **Karma Tracking:** Assigns symbolic value to financial actions.
- **Mentorship:** Offers reflective, growth-oriented advice.
- **Monthly Summary:** Provides a holistic, actionable report.
- **Wellness Scoring:** Aggregates agent outputs into a real-time Financial Wellness Score, with personalized tips.

Each task is context-aware, and the sequential execution ensures that the simulation is both realistic and adaptive.

---

## UI and User Interaction

The system uses Streamlit to present the simulation in an interactive, user-friendly interface. Users can input data, track progress, and receive guidance from all agents, making the experience both accessible and insightful.

---

## Conclusion

This project provides a flexible and modular system for personal finance management, using AI agents to simulate various aspects of financial decision-making in a realistic, context-aware workflow. The monthly simulation logic allows each simulation cycle to build upon the previous one, making the system dynamic and lifelike. All of this is accomplished using free API access by introducing time delays between tasks to avoid rate limit errors. **The best part is that to make the simulator more efficient or to add advanced functionalities, you only need to engineer better prompts or write small Python tool blocks, making the system highly extensible and adaptable to future needs.**
