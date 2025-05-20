# Financial Crew: Migration from CrewAI to LangChain/LangGraph

This document outlines the migration process from CrewAI to LangChain/LangGraph for the Financial Crew simulation project.

## Overview

The Financial Crew project has been transitioned from using the CrewAI framework to LangChain and LangGraph. This migration provides several benefits:

1. **Better Integration**: LangChain and LangGraph are more widely used and have better integration with other tools and frameworks.
2. **More Flexibility**: LangGraph provides more flexibility in defining complex workflows and state management.
3. **Better Debugging**: LangGraph's streaming API allows for better debugging and monitoring of the workflow.
4. **Active Development**: LangChain and LangGraph are actively developed and maintained by a large community.

## Key Files

The migration introduces several new files while preserving the original CrewAI implementation:

- `langgraph_crew.py`: The main LangGraph implementation that replaces `crew.py`
- `api_app_langgraph.py`: FastAPI application using the LangGraph implementation
- `streamlit_app_langgraph.py`: Streamlit application using the LangGraph implementation
- `docs_and_reports/requirements_langchain.txt`: Updated dependencies for LangChain and LangGraph

## Architecture Changes

### From CrewAI to LangGraph

#### CrewAI Architecture (Original)

```
CrewBase
  ├── Agents (defined with @agent decorator)
  ├── Tasks (defined with @task decorator)
  └── Crew (defined with @crew decorator)
      └── Sequential Process
```

#### LangGraph Architecture (New)

```
StateGraph
  ├── Nodes (agent functions)
  ├── Edges (workflow connections)
  └── State (TypedDict for tracking workflow state)
      └── Sequential Execution
```

### Key Differences

1. **State Management**:
   - CrewAI: Implicit state management through task contexts
   - LangGraph: Explicit state management through a TypedDict

2. **Agent Definition**:
   - CrewAI: Agents are defined using decorators and configuration
   - LangGraph: Agents are defined as functions that operate on the state

3. **Workflow Definition**:
   - CrewAI: Workflow is defined by setting task contexts
   - LangGraph: Workflow is defined by adding nodes and edges to a graph

4. **Execution Model**:
   - CrewAI: Tasks are executed sequentially based on context dependencies
   - LangGraph: Nodes are executed based on the graph structure

## Migration Steps

1. **Define State Schema**: Created a `FinancialSimulationState` TypedDict to track the workflow state.

2. **Convert Agents to Nodes**: Converted each CrewAI agent to a LangGraph node function that operates on the state.

3. **Define Graph Structure**: Created a graph with nodes and edges that replicate the sequential workflow of CrewAI.

4. **Implement Simulation Function**: Created a `simulate_timeline_langgraph` function that runs the simulation for multiple months.

5. **Update API and UI**: Updated the FastAPI and Streamlit applications to use the new LangGraph implementation.

## How to Use

### Running with LangGraph

1. Install the new dependencies:
   ```bash
   pip install -r docs_and_reports/requirements_langchain.txt
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app_langgraph.py
   ```

3. Run the API:
   ```bash
   python api_app_langgraph.py
   ```

### Testing the LangGraph Implementation

You can test the LangGraph implementation directly:

```python
from langgraph_crew import simulate_timeline_langgraph

# Test inputs
test_inputs = {
    "user_id": "test_user",
    "user_name": "Test User",
    "age": 30,
    "occupation": "Software Engineer",
    "income_level": "50,000-100,000",
    "goal": "Save $10,000 for emergency fund",
    "starting_balance": 5000,
    "monthly_earning": 6000,
    "monthly_expenses": 4500,
    "savings_target": 1500
}

# Run simulation for 2 months
simulate_timeline_langgraph(2, "Months", test_inputs)
```

## Benefits of the Migration

1. **Improved Modularity**: Each node in the LangGraph workflow is a self-contained function that can be tested independently.

2. **Better Debugging**: The streaming API of LangGraph allows for better debugging and monitoring of the workflow.

3. **More Flexible Workflows**: LangGraph supports more complex workflows, including conditional branching and parallel execution.

4. **Explicit State Management**: The state is explicitly defined and tracked, making it easier to understand and debug the workflow.

5. **Better Integration**: LangChain and LangGraph have better integration with other tools and frameworks in the AI ecosystem.

## Future Improvements

1. **Parallel Execution**: Implement parallel execution of independent nodes to improve performance.

2. **Conditional Workflows**: Add conditional branching to the workflow based on the simulation results.

3. **Better Error Handling**: Implement more robust error handling and recovery mechanisms.

4. **Visualization**: Add visualization of the workflow graph for better understanding and debugging.

5. **LangSmith Integration**: Integrate with LangSmith for better monitoring and debugging of the workflow.

## Conclusion

The migration from CrewAI to LangChain/LangGraph provides a more flexible, maintainable, and powerful implementation of the Financial Crew simulation. The new implementation preserves all the functionality of the original while adding new capabilities and improving the overall architecture.
