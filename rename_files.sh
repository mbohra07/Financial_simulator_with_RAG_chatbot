#!/bin/bash

# Script to rename files to better reflect LangGraph implementation

echo "Renaming files to better reflect LangGraph implementation..."

# Rename crew.py to langgraph_implementation.py
if [ -f "crew.py" ]; then
    mv crew.py langgraph_implementation.py
    echo "Renamed: crew.py -> langgraph_implementation.py"
fi

# Rename api_app.py to langgraph_api.py
if [ -f "api_app.py" ]; then
    mv api_app.py langgraph_api.py
    echo "Renamed: api_app.py -> langgraph_api.py"
fi

# Rename streamlit_app.py to langgraph_streamlit.py
if [ -f "streamlit_app.py" ]; then
    mv streamlit_app.py langgraph_streamlit.py
    echo "Renamed: streamlit_app.py -> langgraph_streamlit.py"
fi

echo "Renaming completed!"
