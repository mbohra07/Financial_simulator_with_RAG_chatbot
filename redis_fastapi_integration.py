"""
Example of integrating Redis with FastAPI in the Financial Crew application.
"""

from fastapi import FastAPI, Depends, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uuid
from datetime import timedelta

# Import Redis utilities
from redis_session import RedisSession
from redis_cache_util import RedisCache
from redis_rate_limiter import RedisRateLimiter

# Create FastAPI app
app = FastAPI(title="Financial Crew API with Redis")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Redis utilities
session_manager = RedisSession()
cache = RedisCache(namespace="financial_api")
rate_limiter = RedisRateLimiter(limit=100, window=60)  # 100 requests per minute


# Pydantic models
class UserInput(BaseModel):
    username: str
    preferences: Dict[str, Any]


class SimulationRequest(BaseModel):
    user_id: str
    parameters: Dict[str, Any]


class SimulationResponse(BaseModel):
    simulation_id: str
    status: str
    results: Optional[Dict[str, Any]] = None


# Dependency for rate limiting
async def check_rate_limit(request: Request, response: Response):
    """Check if the request is rate limited."""
    # Use client IP as the rate limit key
    client_ip = request.client.host
    
    # Check rate limit
    is_limited, remaining, reset_time = rate_limiter.is_rate_limited(client_ip)
    
    # Add rate limit headers to response
    response.headers.update(rate_limiter.get_rate_limit_headers(client_ip, increment=False))
    
    # Raise exception if rate limited
    if is_limited:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Try again in {reset_time} seconds."
        )
    
    return client_ip


# Dependency for session management
async def get_session(request: Request):
    """Get the user session."""
    # Get session ID from cookie or header
    session_id = request.cookies.get("session_id") or request.headers.get("X-Session-ID")
    
    if not session_id:
        # Create new session if none exists
        session_id = str(uuid.uuid4())
        session_data = {"created_at": str(uuid.uuid4())}
        session_manager.set_session(session_id, session_data)
    else:
        # Get existing session
        session_data = session_manager.get_session(session_id)
        if not session_data:
            # Create new session if invalid
            session_id = str(uuid.uuid4())
            session_data = {"created_at": str(uuid.uuid4())}
            session_manager.set_session(session_id, session_data)
    
    return {"id": session_id, "data": session_data}


# Routes
@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Financial Crew API with Redis"}


@app.post("/user/preferences")
async def set_preferences(
    user_input: UserInput,
    session: Dict = Depends(get_session),
    client_ip: str = Depends(check_rate_limit)
):
    """Set user preferences."""
    # Update session with user preferences
    session_data = session["data"]
    session_data["username"] = user_input.username
    session_data["preferences"] = user_input.preferences
    
    # Save updated session
    session_manager.set_session(session["id"], session_data)
    
    # Set session cookie in response
    response = {"message": "Preferences saved", "session_id": session["id"]}
    
    return response


@app.get("/user/preferences")
async def get_preferences(
    session: Dict = Depends(get_session),
    client_ip: str = Depends(check_rate_limit)
):
    """Get user preferences."""
    # Return preferences from session
    session_data = session["data"]
    
    return {
        "username": session_data.get("username"),
        "preferences": session_data.get("preferences", {})
    }


@app.post("/simulation/run", response_model=SimulationResponse)
async def run_simulation(
    simulation_request: SimulationRequest,
    session: Dict = Depends(get_session),
    client_ip: str = Depends(check_rate_limit)
):
    """Run a financial simulation."""
    # Generate simulation ID
    simulation_id = str(uuid.uuid4())
    
    # Store simulation request in cache
    cache_key = f"simulation:{simulation_id}"
    cache.set(
        cache_key,
        {
            "status": "pending",
            "user_id": simulation_request.user_id,
            "parameters": simulation_request.parameters
        },
        expiry=timedelta(hours=1)
    )
    
    # In a real application, you would start the simulation in a background task
    # For this example, we'll just return the simulation ID
    
    return SimulationResponse(
        simulation_id=simulation_id,
        status="pending",
        results=None
    )


@app.get("/simulation/{simulation_id}", response_model=SimulationResponse)
async def get_simulation_result(
    simulation_id: str,
    session: Dict = Depends(get_session),
    client_ip: str = Depends(check_rate_limit)
):
    """Get simulation results."""
    # Get simulation from cache
    cache_key = f"simulation:{simulation_id}"
    simulation_data = cache.get(cache_key)
    
    if not simulation_data:
        raise HTTPException(
            status_code=404,
            detail=f"Simulation with ID {simulation_id} not found"
        )
    
    # In a real application, you would check if the simulation is complete
    # For this example, we'll just return the cached data
    
    return SimulationResponse(
        simulation_id=simulation_id,
        status=simulation_data.get("status", "unknown"),
        results=simulation_data.get("results")
    )


# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
