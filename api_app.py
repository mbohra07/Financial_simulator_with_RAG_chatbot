from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
import json
import os
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functions.crew_functions import simulate_timeline

# ************************************************FastAPI configuration************************************************************
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

def run_simulation_background(task_id: str, user_inputs: dict, simulation_steps: int, simulation_unit: str):
    """Background task to run the simulation"""
    try:
        # Update task status to running
        simulation_tasks[task_id]["status"] = "running"
        
        # Run the simulation with task_id for status updates
        result = simulate_timeline(simulation_steps, simulation_unit, user_inputs, task_id)
        
        # Update task status to completed if not already updated by simulate_timeline
        if task_id in simulation_tasks and simulation_tasks[task_id]["status"] != "completed":
            simulation_tasks[task_id]["status"] = "completed"
            simulation_tasks[task_id]["result"] = result
    except Exception as e:
        # Update task status to failed
        if task_id in simulation_tasks:
            simulation_tasks[task_id]["status"] = "failed"
            simulation_tasks[task_id]["error"] = str(e)
        print(f"❌ Simulation failed: {e}")

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
            2,  # simulation_steps
            "Months"  # simulation_unit
        )
        
        return {
            "status": "success", 
            "message": "Simulation started", 
            "task_id": task_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/check-status/{task_id}")
async def check_simulation_status(task_id: str):
    """Check the status of a simulation task"""
    try:
        if task_id not in simulation_tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Return a copy of the task status to avoid modification during serialization
        task_status = dict(simulation_tasks[task_id])
        # Remove any non-serializable objects
        if "result" in task_status and not isinstance(task_status["result"], (dict, list, str, int, float, bool, type(None))):
            task_status["result"] = str(task_status["result"])
            
        return task_status
    except Exception as e:
        print(f"Error in check_status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-simulation-result/{user_id}")
async def get_simulation_result(user_id: str):
    """Get all simulation results for a user"""
    try:
        output_dir = "output"
        data_dir = "data"
        monthly_output_dir = "monthly_output"
        
        # Standard task files
        task_file_names = [
            "simulated_cashflow",
            "discipline_report",
            "goal_status",
            "behavior_tracker",
            "karmic_tracker",
            "financial_strategy"
        ]
        
        results_payload = {}
        
        # Load standard simulation files
        for file_name in task_file_names:
            dynamic_file_name = f"{user_id}_{file_name}_simulation.json"
            output_path = os.path.join(output_dir, dynamic_file_name)
            if os.path.exists(output_path):
                try:
                    with open(output_path, "r", encoding="utf-8") as f:
                        results_payload[file_name] = json.load(f)
                except Exception as e:
                    results_payload[file_name] = f"Error reading file: {str(e)}"
            else:
                results_payload[file_name] = f"File not found: {output_path}"
        
        # Load persona history
        persona_path = os.path.join(data_dir, "persona_history.json")
        if os.path.exists(persona_path):
            try:
                with open(persona_path, "r", encoding="utf-8") as f:
                    results_payload["persona_history"] = json.load(f)
            except Exception as e:
                results_payload["persona_history"] = f"Error reading persona history: {str(e)}"
        else:
            results_payload["persona_history"] = f"File not found: {persona_path}"
        
        # Load reflection report
        reflection_path = os.path.join(data_dir, "reflection_month.json")
        if os.path.exists(reflection_path):
            try:
                with open(reflection_path, "r", encoding="utf-8") as f:
                    results_payload["reflection_month"] = json.load(f)
            except Exception as e:
                results_payload["reflection_month"] = f"Error reading reflection month: {str(e)}"
        else:
            results_payload["reflection_month"] = f"File not found: {reflection_path}"
                
        return JSONResponse(
            content={
                "status": "success", 
                "user_id": user_id, 
                "results": results_payload
            }
        )
    except Exception as e:
        print(f"Error in get_simulation_result: {str(e)}")
        return JSONResponse(
            status_code=500, 
            content={"status": "error", "message": str(e)}
        )
    
@app.get("/get-simulation-result")
async def get_simulation_result_by_params(user_name: str = Query(...), month: int = Query(...), result: str = Query(...)):
    """
    Handles GET request to receive simulation results.
    This endpoint is used by the simulate_timeline function to push results.
    """
    try:
        parsed_result = json.loads(result)
        print(f"\n📬 Received simulation result for {user_name}, Month {month}")
        return {"status": "success", "user": user_name, "month": month, "data": parsed_result}
    except Exception as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})
    
def main():
    import uvicorn
    uvicorn.run("api_app:app", host="192.168.0.109", port=8000, reload=False)

# If you want to run with `python api_app.py`
if __name__ == "__main__":
    main()
