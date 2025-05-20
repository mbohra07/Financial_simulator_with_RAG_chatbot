"""
FastAPI application for the Financial Crew simulation using LangGraph.
This replaces the CrewAI implementation in api_app.py.
Integrated with MongoDB Atlas for persistent storage and learning from past simulations.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from typing import List
import json
import os
import uuid
from dotenv import load_dotenv
import asyncio
from concurrent.futures import ThreadPoolExecutor
import agentops
from langgraph_implementation import simulate_timeline_langgraph

# Import MongoDB client
from database.mongodb_client import (
    get_all_agent_outputs_for_user,
    get_agent_outputs_for_month
)

# ************************************************FastAPI configuration************************************************************
agentops.init(
     api_key='4be58a32-e415-4142-82b7-834ae6b95422',
     default_tags=['langgraph']
)
app = FastAPI()

# Add CORS middleware to handle OPTIONS requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a thread pool executor for running blocking operations
executor = ThreadPoolExecutor(max_workers=5)

# Store for simulation tasks and their status
simulation_tasks = {}

# Define the expected schema for the incoming JSON
class ExpenseItem(BaseModel):
    name: str
    amount: float

class SimulationInput(BaseModel):
    user_id: str
    user_name: str
    income: float
    expenses: List[ExpenseItem]
    total_expenses: float
    goal: str
    financial_type: str
    risk_level: str

class SimulateRequest(BaseModel):
    n_months: int = 6  # Default to 6 months
    simulation_unit: str = "Months"
    user_inputs: dict

def run_simulation_background(task_id: str, user_inputs: dict, simulation_steps: int, simulation_unit: str):
    """Background task to run the simulation"""
    try:
        # Update task status to running
        simulation_tasks[task_id]["status"] = "running"

        # Run the simulation with task_id for status updates
        result = simulate_timeline_langgraph(simulation_steps, simulation_unit, user_inputs, task_id)

        # Update task status based on result
        if result:
            simulation_tasks[task_id]["status"] = "completed"
        else:
            simulation_tasks[task_id]["status"] = "failed"

    except Exception as e:
        simulation_tasks[task_id]["status"] = "failed"
        simulation_tasks[task_id]["error"] = str(e)
        print(f"Error in simulation task {task_id}: {e}")

@app.post("/start-simulation")
async def start_simulation(payload: SimulationInput, background_tasks: BackgroundTasks):
    """Start a simulation in the background and return a task ID"""
    try:
        # Convert pydantic model to dict
        user_inputs = payload.dict()

        # Generate a unique task ID
        task_id = str(uuid.uuid4())

        # Initialize task status
        simulation_tasks[task_id] = {
            "status": "queued",
            "user_id": user_inputs["user_id"],
            "user_name": user_inputs["user_name"],
            "created_at": asyncio.get_event_loop().time()
        }

        # Run simulation in background using the executor
        loop = asyncio.get_event_loop()
        background_tasks.add_task(
            loop.run_in_executor,
            executor,
            run_simulation_background,
            task_id,
            user_inputs,
            6,  # simulation_steps (6 months)
            "Months"  # simulation_unit
        )

        return {
            "status": "success",
            "message": "Simulation started",
            "task_id": task_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/simulation-status/{task_id}")
async def get_simulation_status(task_id: str):
    """Get the status of a simulation task"""
    if task_id not in simulation_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "status": "success",
        "task_status": simulation_tasks[task_id]["status"],
        "task_details": simulation_tasks[task_id]
    }

@app.post("/simulate")
async def simulate(request: SimulateRequest):
    """Run a simulation directly and return the result"""
    try:
        # Run the simulation
        result = simulate_timeline_langgraph(
            request.n_months,
            request.simulation_unit,
            request.user_inputs
        )

        return {
            "status": "success",
            "message": f"Simulation completed for {request.n_months} {request.simulation_unit}",
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-simulation-result/{user_id}")
async def get_simulation_result(user_id: str):
    """Get all simulation results for a user"""
    try:
        # First try to get data from MongoDB
        mongo_results = get_all_agent_outputs_for_user(user_id)

        if mongo_results:
            # Process MongoDB results
            results = {
                "simulated_cashflow": [],
                "discipline_report": [],
                "goal_status": [],
                "behavior_tracker": [],
                "karmic_tracker": [],
                "financial_strategy": [],
                "reflections": []
            }

            # Group by agent name
            for item in mongo_results:
                agent_name = item.get("agent_name", "")
                month = item.get("month", 0)
                data = item.get("data", {})

                if data:
                    # Add month to data if not present
                    if "month" not in data:
                        data["month"] = month

                    # Add to appropriate category
                    if agent_name == "cashflow" or agent_name == "cashflow_simulator":
                        results["simulated_cashflow"].append(data)
                    elif agent_name == "discipline_tracker":
                        results["discipline_report"].append(data)
                    elif agent_name == "goal_tracker":
                        results["goal_status"].append(data)
                    elif agent_name == "behavior_tracker":
                        results["behavior_tracker"].append(data)
                    elif agent_name == "karma_tracker":
                        results["karmic_tracker"].append(data)
                    elif agent_name == "financial_strategy":
                        results["financial_strategy"].append(data)

            print(f"📊 Retrieved {len(mongo_results)} records from MongoDB for user {user_id}")

            return {
                "status": "success",
                "user_id": user_id,
                "data": results,
                "source": "mongodb"
            }

        # Fallback to file system if MongoDB has no data
        output_dir = "output"
        data_dir = "data"

        # Standard task files
        task_file_names = [
            "simulated_cashflow",
            "discipline_report",
            "goal_status",
            "behavior_tracker",
            "karmic_tracker",
            "financial_strategy"
        ]

        results = {}

        # Collect all task results
        for task_name in task_file_names:
            file_path = f"{output_dir}/{user_id}_{task_name}_simulation.json"
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    results[task_name] = json.load(f)
            else:
                results[task_name] = []

        # Get reflection data if available
        reflection_path = f"{data_dir}/reflection_month.json"
        if os.path.exists(reflection_path):
            with open(reflection_path, "r") as f:
                reflection_data = json.load(f)
                # Filter for this user
                user_reflections = [r for r in reflection_data if r.get("user_name") == user_id]
                results["reflections"] = user_reflections
        else:
            results["reflections"] = []

        print(f"📁 Retrieved data from file system for user {user_id}")

        return {
            "status": "success",
            "user_id": user_id,
            "data": results,
            "source": "filesystem"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

def main():
    import uvicorn
    uvicorn.run("langgraph_api:app", host="0.0.0.0", port=8001, reload=False)

# If you want to run with `python langgraph_api.py`
if __name__ == "__main__":
    main()
