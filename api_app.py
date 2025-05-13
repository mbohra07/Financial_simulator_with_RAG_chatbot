from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from functions.crew_functions import simulate_timeline

# ************************************************Streamlit configuration************************************************************
app = FastAPI()

# Define the expected schema for the incoming JSON
class ExpenseItem(BaseModel):
    name: str
    amount: float

class SimulationInput(BaseModel):
    user_id: int
    user_name: str
    income: float
    expenses: List[ExpenseItem]
    total_expenses: float
    goal: str
    financial_type: str
    risk_level: str
    balance: float



@app.post("/simulate")
async def run_simulation(payload: SimulationInput):
    try:
        # Convert pydantic model to dict and adjust for simulate_timeline
        user_inputs = payload.dict()
        # You can adjust these as needed, or accept them via the API as well
        simulation_steps = 6
        simulation_unit = "Months"
        
        # Run your simulation
        result = simulate_timeline(simulation_steps, simulation_unit, user_inputs)
        if result:
            return {"status": "success", "details": "Simulation complete", "result": result}
        else:
            raise HTTPException(status_code=500, detail="Simulation failed after multiple attempts.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


# Optional: main function for programmatic start (not required for uvicorn CLI)
def main():
    import uvicorn
    uvicorn.run("fastapi_simulation_api:app", host="192.168.0.109", port=8000, reload=True)

# If you want to run with `python fastapi_simulation_api.py`
if __name__ == "__main__":
    main()