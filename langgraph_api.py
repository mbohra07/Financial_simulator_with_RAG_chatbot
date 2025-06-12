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
import uvicorn
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
from database.mongodb_client import save_chat_message

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

# Store for teacher agent tasks and their status
teacher_tasks = {}

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
    simulation_id: Optional[str] = None  # Optional simulation ID

# Teacher agent models
class TeacherQuery(BaseModel):
    user_id: str
    query: str
    pdf_id: Optional[Union[str, List[str]]] = None  # Optional PDF ID(s) to search in specific PDF(s)
    wait: Optional[bool] = False  # Whether to wait for the response or return immediately

class TeacherResponse(BaseModel):
    """Response model for the teacher agent endpoint"""
    response: str
    chat_history: Optional[List[Dict[str, str]]] = None
    learning_task_id: Optional[str] = None  # Only use learning_task_id for clarity
    status: Optional[str] = None

def run_teacher_agent_background(task_id: str, user_id: str, query: str, chat_history: List[Dict[str, str]], pdf_id: Union[str, List[str], None] = None):
    """Background task to run the teacher agent with MongoDB Atlas Vector Search"""
    try:
        # Update task status to running
        teacher_tasks[task_id]["status"] = "running"

        print(f"🚀 Running teacher agent in background - task_id: {task_id}, user_id: {user_id}")

        # Check if MongoDB is available
        from database.mongodb_client import USE_MOCK_DB
        if USE_MOCK_DB:
            error_msg = "MongoDB Atlas connection is required for vector search functionality."
            error_message = f"I'm sorry, but MongoDB Atlas is required to process your question. Please contact the administrator."

            teacher_tasks[task_id]["status"] = "failed"
            teacher_tasks[task_id]["error"] = error_msg
            teacher_tasks[task_id]["response"] = error_message
            teacher_tasks[task_id]["chat_history"] = chat_history + [
                {"role": "user", "content": query},
                {"role": "assistant", "content": error_message}
            ]

            # Save the error message to the database
            try:
                save_chat_message(user_id, "user", query)
                save_chat_message(user_id, "assistant", error_message)
            except Exception:
                pass  # Ignore database errors at this point

            print(f"❌ {error_msg}")
            return

        try:
            # Run the teacher agent
            result = run_teacher_agent(
                user_query=query,
                user_id=user_id,
                chat_history=chat_history,
                pdf_id=pdf_id
            )

            # Ensure we have a valid response
            response = result.get("response")
            if not response:
                response = "I'm sorry, I couldn't generate a response to your question."
                result["response"] = response

            # Save the response and chat history
            teacher_tasks[task_id]["response"] = response
            teacher_tasks[task_id]["chat_history"] = result.get("chat_history", [])

            # Save the new messages to the database
            save_chat_message(user_id, "user", query)
            save_chat_message(user_id, "assistant", response)

            # Update task status
            teacher_tasks[task_id]["status"] = "completed"
            print(f"✅ Teacher agent task completed - task_id: {task_id}")

        except ValueError as ve:
            if "MongoDB" in str(ve):
                error_msg = str(ve)
                error_message = f"I'm sorry, but MongoDB Atlas is required to process your question. Please contact the administrator."

                teacher_tasks[task_id]["status"] = "failed"
                teacher_tasks[task_id]["error"] = error_msg
                teacher_tasks[task_id]["response"] = error_message
                teacher_tasks[task_id]["chat_history"] = chat_history + [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": error_message}
                ]

                # Save the error message to the database
                try:
                    save_chat_message(user_id, "user", query)
                    save_chat_message(user_id, "assistant", error_message)
                except Exception:
                    pass  # Ignore database errors at this point

                print(f"❌ {error_msg}")
            else:
                raise

    except Exception as e:
        error_message = f"I'm sorry, but there was an error processing your question: {str(e)}"
        teacher_tasks[task_id]["status"] = "failed"
        teacher_tasks[task_id]["error"] = str(e)
        teacher_tasks[task_id]["response"] = error_message
        teacher_tasks[task_id]["chat_history"] = chat_history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": error_message}
        ]

        # Save the error message to the database
        try:
            save_chat_message(user_id, "user", query)
            save_chat_message(user_id, "assistant", error_message)
        except Exception:
            pass  # Ignore database errors at this point

        print(f"❌ Error in teacher agent task {task_id}: {e}")
        import traceback
        traceback.print_exc()

def run_simulation_background(task_id: str, user_inputs: dict, simulation_steps: int, simulation_unit: str):
    """Background task to run the simulation"""
    try:
        # Update task status to running
        simulation_tasks[task_id]["status"] = "running"

        # Generate a simulation ID
        from database.mongodb_client import (
            save_user_input,
            save_agent_output,
            get_previous_month_outputs,
            get_agent_outputs_for_month,
            generate_simulation_id
        )

        # Initialize MongoDB collections
        pdf_collection = get_database()["pdf_metadata"]
        chunks_collection = get_database()["pdf_chunks"]

        # Store simulation_id in task details
        simulation_id = generate_simulation_id()
        simulation_tasks[task_id]["simulation_id"] = simulation_id
        print(f"📝 Simulation ID: {simulation_id} for task {task_id}")

        # Run the simulation with task_id and simulation_id for status updates
        result = simulate_timeline_langgraph(
            simulation_steps,
            simulation_unit,
            user_inputs,
            task_id,
            simulation_id
        )

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

        # Get user_id
        user_id = user_inputs["user_id"]

        # Clear previous simulation data for this user (optional)
        try:
            # Get the database
            from database.mongodb_client import get_database
            db = get_database()

            if db:
                # Delete previous agent outputs for this user
                collection = db["agent_outputs"]
                delete_result = collection.delete_many({"user_id": user_id})
                print(f"🧹 Deleted {delete_result.deleted_count} previous simulation records for user {user_id}")
        except Exception as e:
            print(f"⚠️ Warning: Could not clear previous simulation data: {e}")

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

@app.get("/simulation-results/{task_id}")
async def get_simulation_results(task_id: str):
    """Get the latest results for a simulation task in progress"""
    if task_id not in simulation_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    # Get user_id from the task
    user_id = simulation_tasks[task_id].get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in task")

    # Get simulation_id if available
    simulation_id = simulation_tasks[task_id].get("simulation_id")

    # Get the latest results from MongoDB
    try:
        # If simulation_id is available, filter by it for more precise results
        if simulation_id:
            # Get the database
            from database.mongodb_client import get_database
            db = get_database()

            # Query for this specific simulation
            query = {
                "user_id": user_id,
                "simulation_id": simulation_id
            }

            # Get agent outputs collection
            collection = db["agent_outputs"]
            mongo_results = list(collection.find(query))

            # Convert ObjectId to string
            for result in mongo_results:
                if "_id" in result:
                    result["_id"] = str(result["_id"])
        else:
            # Fall back to getting all results for the user
            from database.mongodb_client import get_all_agent_outputs_for_user
            mongo_results = get_all_agent_outputs_for_user(user_id)

        # Process results into categories
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

            # Remove the "reflections" key if it exists
            if "reflections" in results:
                del results["reflections"]

            # Add person_history from data folder with user_id prefix
            data_dir = "data"

            # List all files in the data directory to debug
            print(f"📂 Files in {data_dir} directory:")
            try:
                for file in os.listdir(data_dir):
                    print(f"  - {file}")
            except Exception as e:
                print(f"⚠️ Error listing files in {data_dir}: {e}")

            # Try both naming patterns for person_history
            person_history_path = f"{data_dir}/{user_id}_person_history.json"
            alt_person_history_path = f"{data_dir}/string_persona_history.json"  # The actual file name from logs

            print(f"🔍 Looking for person_history at: {person_history_path}")
            if os.path.exists(person_history_path):
                print(f"✅ Found person_history file with user_id prefix: {person_history_path}")
                try:
                    with open(person_history_path, "r") as f:
                        person_history_data = json.load(f)
                        results["person_history"] = person_history_data
                        print(f"📊 Loaded {len(person_history_data) if isinstance(person_history_data, list) else 1} person_history entries")
                except Exception as e:
                    print(f"⚠️ Warning: Could not load {user_id}_person_history.json: {e}")
                    results["person_history"] = []
            elif os.path.exists(alt_person_history_path):
                print(f"✅ Found person_history file with alternate name: {alt_person_history_path}")
                try:
                    with open(alt_person_history_path, "r") as f:
                        person_history_data = json.load(f)
                        results["person_history"] = person_history_data
                        print(f"📊 Loaded {len(person_history_data) if isinstance(person_history_data, list) else 1} person_history entries from alternate file")
                except Exception as e:
                    print(f"⚠️ Warning: Could not load {alt_person_history_path}: {e}")
                    results["person_history"] = []
            else:
                print(f"❌ Person history file with user_id prefix not found")
                # Try without user_id prefix as fallback
                fallback_path = f"{data_dir}/person_history.json"
                print(f"🔍 Looking for fallback person_history at: {fallback_path}")
                if os.path.exists(fallback_path):
                    print(f"✅ Found fallback person_history file: {fallback_path}")
                    try:
                        with open(fallback_path, "r") as f:
                            person_history_data = json.load(f)
                            results["person_history"] = person_history_data
                            print(f"📊 Loaded {len(person_history_data) if isinstance(person_history_data, list) else 1} person_history entries from fallback")
                    except Exception as e:
                        print(f"⚠️ Warning: Could not load fallback person_history.json: {e}")
                        results["person_history"] = []
                else:
                    print(f"❌ Fallback person_history file not found")
                    results["person_history"] = []

            # Add monthly reflections from monthly_output folder
            monthly_output_dir = "monthly_output"
            monthly_reflections = []

            # Create monthly_output directory if it doesn't exist
            os.makedirs(monthly_output_dir, exist_ok=True)
            print(f"📁 Monthly output directory: {monthly_output_dir}")

            # Get all months from the simulation results
            months = set()
            for category in ["simulated_cashflow", "discipline_report", "goal_status", "behavior_tracker", "karmic_tracker", "financial_strategy"]:
                for item in results.get(category, []):
                    if "month" in item:
                        months.add(item["month"])

            print(f"🔢 Found months in simulation results: {months}")

            # Load reflection files for each month with user_id prefix
            for month in months:
                reflection_path = f"{monthly_output_dir}/{user_id}_reflection_month_{month}.json"
                print(f"🔍 Looking for monthly reflection at: {reflection_path}")
                if os.path.exists(reflection_path):
                    print(f"✅ Found monthly reflection file with user_id prefix: {reflection_path}")
                    try:
                        with open(reflection_path, "r") as f:
                            reflection_data = json.load(f)
                            # Add month to the reflection data
                            if isinstance(reflection_data, dict):
                                reflection_data["month"] = month
                            monthly_reflections.append(reflection_data)
                            print(f"📊 Loaded monthly reflection for month {month}")
                    except Exception as e:
                        print(f"⚠️ Warning: Could not load {user_id}_reflection_month_{month}.json: {e}")

                        # Try without user_id prefix as fallback
                        fallback_path = f"{monthly_output_dir}/reflection_month_{month}.json"
                        print(f"🔍 Looking for fallback monthly reflection at: {fallback_path}")
                        if os.path.exists(fallback_path):
                            print(f"✅ Found fallback monthly reflection file: {fallback_path}")
                            try:
                                with open(fallback_path, "r") as f:
                                    reflection_data = json.load(f)
                                    # Add month to the reflection data
                                    if isinstance(reflection_data, dict):
                                        reflection_data["month"] = month
                                    monthly_reflections.append(reflection_data)
                                    print(f"📊 Loaded monthly reflection for month {month} from fallback")
                            except Exception as e:
                                print(f"⚠️ Warning: Could not load fallback reflection_month_{month}.json: {e}")
                        else:
                            print(f"❌ Fallback monthly reflection file for month {month} not found")
                else:
                    print(f"❌ Monthly reflection file with user_id prefix for month {month} not found")

            # Add monthly reflections to results
            results["monthly_reflections"] = monthly_reflections
            print(f"📊 Added {len(monthly_reflections)} monthly reflections to results")

            return {
                "status": "success",
                "task_id": task_id,
                "task_status": simulation_tasks[task_id]["status"],
                "user_id": user_id,
                "data": results,
                "source": "mongodb"
            }
        else:
            # No results yet
            return {
                "status": "success",
                "task_id": task_id,
                "task_status": simulation_tasks[task_id]["status"],
                "user_id": user_id,
                "data": {
                    "simulated_cashflow": [],
                    "discipline_report": [],
                    "goal_status": [],
                    "behavior_tracker": [],
                    "karmic_tracker": [],
                    "financial_strategy": [],
                    "person_history": [],
                    "monthly_reflections": []
                },
                "message": "No simulation results available yet"
            }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error retrieving simulation results: {str(e)}"
            }
        )

# /simulate endpoint removed as it's not being used

# /get-simulation-result/{user_id} endpoint removed as it's not being used

# Teacher agent endpoints
@app.post("/user/learning", response_model=TeacherResponse)
async def learning_endpoint(
    query: TeacherQuery,
    background_tasks: BackgroundTasks
):
    """Process a learning query from the user and return a response from the teacher agent using MongoDB Atlas Vector Search"""
    try:
        actual_user_id = query.user_id
        actual_query = query.query
        actual_pdf_id = query.pdf_id
        actual_wait = query.wait

        print(f"📥 Received learning request - user_id: {actual_user_id}, query: '{actual_query}'")

        # Check if MongoDB is available
        from database.mongodb_client import USE_MOCK_DB
        if USE_MOCK_DB:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "MongoDB Atlas connection is required for vector search functionality.",
                    "requires_mongodb_atlas": True
                }
            )

        # Get chat history from database but limit to last 10 messages to avoid overwhelming context
        from database.mongodb_client import get_chat_history_for_user
        chat_history = get_chat_history_for_user(actual_user_id, limit=10)

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
        if actual_pdf_id:
            if isinstance(actual_pdf_id, list):
                # If multiple PDF IDs are provided
                print(f"📚 Using multiple PDFs: {', '.join(actual_pdf_id)}")
                if len(actual_pdf_id) > 0:
                    pdf_id_param = actual_pdf_id
            else:
                # Single PDF ID
                print(f"📚 Using specific PDF: {actual_pdf_id}")
                pdf_id_param = actual_pdf_id

        # Ensure query is properly formatted
        current_query = actual_query.strip()
        print(f"🔍 Processing query: '{current_query}'")

        # Generate a unique task ID
        task_id = str(uuid.uuid4())

        # Check if we should wait for the response or run in the background
        if actual_wait:
            # Run the teacher agent synchronously
            print(f"⏱️ Running teacher agent synchronously - task_id: {task_id}")
            try:
                result = run_teacher_agent(
                    user_query=current_query,
                    user_id=actual_user_id,
                    chat_history=formatted_chat_history,
                    pdf_id=pdf_id_param
                )

                # Log the response
                print(f"✅ Teacher agent response: '{result['response'][:50]}...'")

                # Save the new messages to the database
                save_chat_message(actual_user_id, "user", current_query)
                save_chat_message(actual_user_id, "assistant", result["response"])

                return TeacherResponse(
                    response=result["response"],
                    chat_history=result["chat_history"],
                    learning_task_id=task_id,  # Only use learning_task_id for clarity
                    status="completed"
                )
            except ValueError as ve:
                if "MongoDB" in str(ve):
                    return JSONResponse(
                        status_code=400,
                        content={
                            "status": "error",
                            "message": str(ve),
                            "requires_mongodb_atlas": True
                        }
                    )
                else:
                    raise
        else:
            # Run the teacher agent in the background
            print(f"🔄 Running teacher agent in background - task_id: {task_id}")

            # Initialize task status
            teacher_tasks[task_id] = {
                "status": "queued",
                "user_id": actual_user_id,
                "query": current_query,
                "created_at": asyncio.get_event_loop().time(),
                "response": None,
                "chat_history": None
            }

            # Run teacher agent in background using the executor
            loop = asyncio.get_event_loop()
            background_tasks.add_task(
                loop.run_in_executor,
                executor,
                run_teacher_agent_background,
                task_id,
                actual_user_id,
                current_query,
                formatted_chat_history,
                pdf_id_param
            )

            # Save the user message to the database immediately
            save_chat_message(actual_user_id, "user", current_query)

            return TeacherResponse(
                response="Your question is being processed. Please check back in a moment for the response.",
                learning_task_id=task_id,  # Only use learning_task_id for clarity
                status="queued"
            )
    except Exception as e:
        print(f"❌ Error in learning endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/learning/{learning_task_id}", response_model=TeacherResponse)
async def get_learning_status(learning_task_id: str):
    """Get the status and response of a teacher agent task"""
    if learning_task_id not in teacher_tasks:
        raise HTTPException(status_code=404, detail="Learning task not found")

    task = teacher_tasks[learning_task_id]
    status = task.get("status", "queued")

    # Ensure we have a valid response string
    response = task.get("response")
    if response is None:
        if status == "failed":
            response = "I'm sorry, but there was an error processing your question. Please try again."
        else:
            response = "Your question is still being processed. Please check back in a moment."

    # Return the response with all required fields
    return TeacherResponse(
        response=response,
        chat_history=task.get("chat_history", []),
        learning_task_id=learning_task_id,  # Only use learning_task_id for clarity
        status=status
    )

@app.post("/pdf/chat")
async def pdf_upload_endpoint(
    user_id: str = Form(...),
    pdf_file: UploadFile = File(...)
):
    """Upload a PDF file for the teacher agent to use in explanations with MongoDB Atlas Vector Search"""
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
            # Return a more specific error message for MongoDB Atlas requirement
            if "MongoDB Atlas" in result["message"]:
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "message": result["message"],
                        "requires_mongodb_atlas": True
                    }
                )
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
        pdf_id: Optional PDF ID or list of PDF IDs to remove specific PDFs
    """
    try:
        print(f"🗑️ PDF removal request - user_id: {request.user_id}, pdf_id: {request.pdf_id}")

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
            "pdfs": pdf_docs
        }
    except Exception as e:
        print(f"❌ Error listing PDFs: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def main():
    try:
        uvicorn.run("langgraph_api:app",host="192.168.3.104",port=8000,reload=False)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()
        raise

# If you want to run with `python langgraph_api.py`
if __name__ == "__main__":
    main()
