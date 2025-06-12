# Financial Simulator with RAG Chatbot

A financial simulation system with RAG (Retrieval-Augmented Generation) chatbot capabilities using LangChain and FastAPI.

## Features

- Financial simulation over multiple months
- RAG-based chatbot for financial knowledge
- PDF document processing and retrieval
- MongoDB Atlas integration for persistent storage
- Agent-based simulation system
- Real-time simulation progress tracking

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with the following variables:
```
MONGODB_URI=your_mongodb_uri
OPENAI_API_KEY=your_openai_api_key
```

4. Run the server:
```bash
python langgraph_api.py
```

The server will be available at http://192.168.3.104:8000

## API Endpoints

- `/start-simulation`: Start a new financial simulation
- `/pdf/upload`: Upload PDF documents for RAG
- `/pdf/list`: List uploaded PDF documents
- `/learning`: Get financial knowledge from chatbot
- `/simulation-status`: Check simulation status

## Project Structure

```
financial_crew/
├── config/
│   ├── agents.yaml
│   └── tasks.yaml
├── database/
│   ├── __init__.py
│   └── mongodb_client.py
├── functions/
│   ├── crew_functions.py
│   ├── economic_context.py
│   ├── kickoff_functions.py
│   ├── monthly_simulation.py
│   ├── streamlit_functions.py
│   ├── task_functions.py
│   └── task_functions_fixed.py
├── data/
│   ├── financial_concepts.json
│   ├── financial_knowledge.json
│   └── user_profile.yaml
├── temp_pdfs/
├── monthly_output/
├── output/
├── utils/
│   ├── __init__.py
│   └── json_fix.py
├── langgraph_api.py
├── langgraph_implementation.py
├── langgraph_streamlit.py
├── requirements.txt
├── .env
└── README.md
```

## Running the Streamlit Interface

To run the Streamlit interface:
```bash
streamlit run langgraph_streamlit.py
```

## MongoDB Collections

The application uses MongoDB for persistent storage with the following collections:
- `user_inputs`: Stores user simulation inputs
- `agent_outputs`: Stores simulation results
- `chat_history`: Stores chat interactions
- `pdf_metadata`: Stores PDF document information
- `pdf_chunks`: Stores PDF content chunks for RAG

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
