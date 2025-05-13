# src/financial_crew/crew.py
from crewai import Agent, Crew, Process, Task, LLM, TaskOutput, Flow
from langchain_groq import ChatGroq
from crewai.project import CrewBase, agent, crew, task
import openai
from dotenv import load_dotenv
import os
import numpy as np
import json
import litellm  
import time
import asyncio
from functools import wraps
from langchain_openai import ChatOpenAI
from Tools.financial_tools import (opportunity_cost_analyzer,budget_allocator,utility_evaluator,debt_repayment_tool,portfolio_simulation_tool,emergency_fund_tool,rl_lite_tool,karma_tracker_tool)
load_dotenv()

# ***********************************************Initialising LLM API to run my agent workflow************************************************************************
groq_llm = LLM(model="groq/llama-3.1-70b-versatile")

#litellm.provider = "ollama"
#litellm.model = "deepseek-r1:7b"
#litellm.api_base = "http://localhost:11434

# *************************************************************Crew Base**************************************************************
@CrewBase
class FinancialCrew():

    def __init__(self):
        # Required Crew attributes
        self.name = "Financial Simulation Crew"
        self.description = "Manages monthly financial analysis and decision making"
        self.verbose = True
    # **********************************************************Agents****************************************************************
    
    @agent
    def emotional_bias_agent(self) -> Agent:
        return Agent(config=self.agents_config['emotional_bias_agent'], verbose=True, allow_delegation=True)
    
    @agent
    def goal_tracker_agent(self) -> Agent:
        return Agent(config=self.agents_config['goal_tracker_agent'], verbose=True, allow_delegation=True)

    @agent
    def financial_strategy_agent(self) -> Agent:
        return Agent(config=self.agents_config['financial_strategy_agent'],  verbose=True, allow_delegation=True)
    
    @agent
    def behavior_tracker_agent(self) -> Agent:
        return Agent(config=self.agents_config['behavior_tracker_agent'], verbose=True,allow_delegation=True)
    
    @agent
    def discipline_tracker_agent(self) -> Agent:
        return Agent(config=self.agents_config['discipline_tracker_agent'], verbose=True, allow_delegation=True)
    
    @agent
    def karma_tracker_agent(self) -> Agent:
        return Agent(config=self.agents_config['karma_tracker_agent'], verbose=True, allow_delegation=True)
    
    # **********************************************************Tasks*******************************************************************
    @task
    def simulate_cashflow_task(self) -> Task:
        return Task(config=self.tasks_config['simulate_cashflow_task'])
    
    @task
    def discipline_tracker_task(self) -> Task:
        return Task(config=self.tasks_config['discipline_tracker_task'])
    
    @task
    def track_goals(self) -> Task:
        return Task(config=self.tasks_config['track_goals'])
    
    @task
    def behavior_tracker_task(self) -> Task:
        return Task(config=self.tasks_config['behavior_tracker_task'])
    
    @task
    def karma_tracker_task(self) -> Task:
        return Task(config=self.tasks_config['karma_tracker_task'])
    
    @task
    def financial_strategy_task(self) -> Task:
        return Task(config=self.tasks_config['financial_strategy_task'])

    @crew
    def flexible_crew(self, input_data=None, agent_overrides=None, task_overrides=None) -> Crew:
        # ***************************************************Override agents and tasks if necessary**************************************
        agents = self.agents
        if agent_overrides:
            for agent_name, override_config in agent_overrides.items():
                agent = getattr(self, agent_name)()
                agent.config.update(override_config)
                agents[agent_name] = agent
        
        tasks = self.tasks
        if task_overrides:
            for task_name, override_config in task_overrides.items():
                task = getattr(self, task_name)()
                task.config.update(override_config)
                tasks[task_name] = task

        # ****************************************************Create task context dependencies********************************************
        simulate_task = self.simulate_cashflow_task()

        discipline_tracker_task = self.discipline_tracker_task()
        goal_task = self.track_goals()
        behaviour_tracker_task = self.behavior_tracker_task()
        karma_tracker_task = self.karma_tracker_task()
        financial_strategy_task = self.financial_strategy_task()

        # Apply the correct context chain

        discipline_tracker_task.context = [simulate_task]

        goal_task.context = [simulate_task, discipline_tracker_task]

        behaviour_tracker_task.context = [simulate_task, discipline_tracker_task, goal_task]

        karma_tracker_task.context = [simulate_task, discipline_tracker_task, goal_task, behaviour_tracker_task]

        financial_strategy_task.context = [simulate_task, discipline_tracker_task, goal_task, behaviour_tracker_task, karma_tracker_task]

        # ******************************************************Create and return the crew***********************************************
        return Crew(
            agents=agents,
            tasks=tasks,
            input_data=input_data,
            process=Process.sequential,
            verbose=True,
            memory=False
        )