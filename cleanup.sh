#!/bin/bash

# Script to clean up unnecessary files from the codebase

# Confirm before proceeding
echo "This script will remove unnecessary files from your codebase."
echo "The following files and directories will be removed:"
echo "- Duplicate files from the migration (if you're fully transitioning)"
echo "- Training-related files (if you're not actively training models)"
echo "- Empty or unused directories"
echo "- Log files"
echo ""
echo "Do you want to proceed? (y/n)"
read -r response

if [[ "$response" != "y" ]]; then
    echo "Operation cancelled."
    exit 0
fi

# Remove duplicate files from the migration
echo "Do you want to keep the LangGraph implementation and remove the CrewAI implementation? (y/n)"
read -r keep_langgraph

if [[ "$keep_langgraph" == "y" ]]; then
    echo "Removing CrewAI implementation files..."
    rm -f crew.py
    rm -f api_app.py
    rm -f streamlit_app.py
    # Rename LangGraph files to standard names
    mv langgraph_crew.py crew.py
    mv api_app_langgraph.py api_app.py
    mv streamlit_app_langgraph.py streamlit_app.py
else
    echo "Removing LangGraph implementation files..."
    rm -f langgraph_crew.py
    rm -f api_app_langgraph.py
    rm -f streamlit_app_langgraph.py
fi

# Remove training-related files
echo "Do you want to remove training-related files? (y/n)"
read -r remove_training

if [[ "$remove_training" == "y" ]]; then
    echo "Removing training-related files..."
    rm -f train_model.py
    rm -f training_data.pkl
    rm -f your_model.pkl
fi

# Remove empty or unused directories
echo "Do you want to remove empty or unused directories? (y/n)"
read -r remove_dirs

if [[ "$remove_dirs" == "y" ]]; then
    echo "Removing empty or unused directories..."
    # Check if directories are empty before removing
    if [ -z "$(ls -A monthly_output)" ]; then
        rm -rf monthly_output
    else
        echo "monthly_output is not empty, skipping..."
    fi
    
    if [ -z "$(ls -A output)" ]; then
        rm -rf output
    else
        echo "output is not empty, skipping..."
    fi
fi

# Remove log files
echo "Do you want to remove log files? (y/n)"
read -r remove_logs

if [[ "$remove_logs" == "y" ]]; then
    echo "Removing log files..."
    rm -f agentops.log
fi

echo "Cleanup completed!"
