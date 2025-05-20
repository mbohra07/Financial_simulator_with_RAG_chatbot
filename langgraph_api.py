"""
FastAPI application for the Financial Crew simulation using LangGraph.
This replaces the CrewAI implementation in api_app.py.
Integrated with MongoDB Atlas for persistent storage and learning from past simulations.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional, Union
import json
import os
import uuid
import shutil
from pathlib import Path
from dotenv import load_dotenv
import asyncio
from concurrent.futures import ThreadPoolExecutor
import agentops
from langgraph_implementation import simulate_timeline_langgraph
from teacher_agent import run_teacher_agent, handle_pdf_upload, handle_pdf_removal

# Import MongoDB client
from database.mongodb_client import (
    get_all_agent_outputs_for_user,
    get_agent_outputs_for_month,
    save_chat_message,
    get_chat_history_for_user
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

# Teacher agent models
class TeacherQuery(BaseModel):
    user_id: str
    query: str
    pdf_id: Optional[Union[str, List[str]]] = None  # Optional PDF ID(s) to search in specific PDF(s)

class TeacherResponse(BaseModel):
    """Response model for the teacher agent endpoint"""
    response: str

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
        user_inputs = payload.model_dump()

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

# Teacher agent endpoints
@app.post("/user/learning", response_model=TeacherResponse)
async def learning_endpoint(query: TeacherQuery):
    """Process a learning query from the user and return a response from the teacher agent"""
    try:
        print(f"📥 Received learning request - user_id: {query.user_id}, query: '{query.query}'")

        # Get chat history from database but limit to last 10 messages to avoid overwhelming context
        chat_history = get_chat_history_for_user(query.user_id, limit=10)

        # Convert to the format expected by the teacher agent
        formatted_chat_history = []
        for msg in chat_history:
            formatted_chat_history.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })

        print(f"📜 Retrieved {len(formatted_chat_history)} chat history messages")
        if formatted_chat_history:
            print(f"📜 Most recent message - Role: {formatted_chat_history[-1].get('role')}, Content: {formatted_chat_history[-1].get('content', '')[:50]}...")

        # Handle PDF IDs
        pdf_id_param = None
        if query.pdf_id:
            if isinstance(query.pdf_id, list):
                # If multiple PDF IDs are provided
                print(f"📚 Using multiple PDFs: {', '.join(query.pdf_id)}")
                if len(query.pdf_id) > 0:
                    pdf_id_param = query.pdf_id
            else:
                # Single PDF ID
                print(f"📚 Using specific PDF: {query.pdf_id}")
                pdf_id_param = query.pdf_id

        # Ensure query is properly formatted
        current_query = query.query.strip()
        print(f"🔍 Processing query: '{current_query}'")

        # Run the teacher agent with fresh state
        result = run_teacher_agent(
            user_query=current_query,
            user_id=query.user_id,
            chat_history=formatted_chat_history,
            pdf_id=pdf_id_param
        )

        # Log the response
        print(f"✅ Teacher agent response: '{result['response'][:50]}...'")

        # Save the new messages to the database
        save_chat_message(query.user_id, "user", current_query)
        save_chat_message(query.user_id, "assistant", result["response"])

        return TeacherResponse(
            response=result["response"]
        )
    except Exception as e:
        print(f"❌ Error in learning endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pdf/chat")
async def pdf_upload_endpoint(
    user_id: str = Form(...),
    pdf_file: UploadFile = File(...)
):
    """Upload a PDF file for the teacher agent to use in explanations"""
    try:
        # Create temp directory if it doesn't exist
        temp_dir = Path("temp_pdfs")
        temp_dir.mkdir(exist_ok=True)

        # Save the uploaded file
        file_path = temp_dir / f"{user_id}_{pdf_file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(pdf_file.file, buffer)

        # Process the PDF
        result = handle_pdf_upload(str(file_path), user_id)

        if result["success"]:
            return {
                "status": "success",
                "message": result["message"],
                "user_id": user_id,
                "pdf_id": result["pdf_id"],
                "chunk_count": result.get("chunk_count", 0)
            }
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": result["message"]
                }
            )
    except Exception as e:
        print(f"❌ Error in PDF upload: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Define a model for PDF removal request
class PDFRemovalRequest(BaseModel):
    user_id: str
    pdf_id: Optional[Union[str, List[str]]] = None

@app.post("/pdf/removed")
async def pdf_removal_endpoint(request: PDFRemovalRequest):
    """
    Remove PDF data for a user or a specific PDF

    Request body:
        user_id: User identifier
        pdf_id: Optional PDF ID to remove a specific PDF
    """
    try:
        # Remove PDF data
        result = handle_pdf_removal(request.user_id, request.pdf_id)

        if result["success"]:
            return {
                "status": "success",
                "message": result["message"],
                "user_id": request.user_id,
                "pdf_id": result.get("pdf_id")
            }
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": result["message"]
                }
            )
    except Exception as e:
        print(f"❌ Error in PDF removal: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pdf/list")
async def pdf_list_endpoint(user_id: str):
    """List all PDFs for a user"""
    try:
        # Get MongoDB database
        from database.mongodb_client import get_database
        db = get_database()

        # Get all PDFs for this user
        pdf_docs = list(db["pdf_metadata"].find({"user_id": user_id}))

        # Convert ObjectId to string
        for doc in pdf_docs:
            if "_id" in doc:
                doc["_id"] = str(doc["_id"])

        return {
            "status": "success",
            "user_id": user_id,
            "pdf_count": len(pdf_docs),
            "pdfs": pdf_docs
        }
    except Exception as e:
        print(f"❌ Error listing PDFs: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def main():
    import uvicorn
    uvicorn.run("langgraph_api:app", host="0.0.0.0", port=8001, reload=False)

# If you want to run with `python langgraph_api.py`
if __name__ == "__main__":
    main()
