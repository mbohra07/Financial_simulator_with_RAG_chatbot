"""
Database package for the Financial Simulation system.
"""

from database.mongodb_client import (
    get_client,
    get_database,
    get_user_inputs_collection,
    get_agent_outputs_collection,
    close_connection,
    save_user_input,
    save_agent_output,
    get_agent_outputs_for_month,
    get_previous_month_outputs,
    get_all_agent_outputs_for_user,
    generate_simulation_id
)

__all__ = [
    "get_client",
    "get_database",
    "get_user_inputs_collection",
    "get_agent_outputs_collection",
    "close_connection",
    "save_user_input",
    "save_agent_output",
    "get_agent_outputs_for_month",
    "get_previous_month_outputs",
    "get_all_agent_outputs_for_user",
    "generate_simulation_id"
]
