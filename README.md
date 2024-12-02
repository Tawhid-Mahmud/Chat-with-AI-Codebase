# AI Chatbot for Codebase

A Streamlit-based AI chatbot that combines vector search capabilities with Pinecone and LLM integration using Groq. The application allows users to search through codebases and get AI-powered explanations.

## Features

- 🔍 Vector-based code search using Pinecone
- 🤖 AI-powered code explanations using Groq LLM
- 💬 Interactive chat interface
- 🔄 Multiple LLM model support
- 📊 Semantic search capabilities

## Installation

1. Clone the repository:
2. Install required packages:
    pip install -r requirements.txt
4. Set up environment variables:
    PINECONE_API_KEY=your_pinecone_api_key
    GROQ_API_KEY=your_groq_api_key
    PINECONE_HOST=your_pinecone_host
5. Run the Streamlit app:
    

6. Open your browser and navigate to `http://localhost:8501`

7. Use the interface to:
   - Search through your codebase
   - Chat with the AI assistant
   - Get code explanations

## Configuration

- Supported LLM models:
  - llama-3.1-70b-versatile
  - grok-model-2
  - grok-model-3

- Vector search is configured with:
  - Dimension: 768
  - Metric: cosine
  - Model: all-MiniLM-L6-v2

## Dependencies
streamlit
gitpython
pinecone-client
groq
sentence-transformers
PyGithub
langchain-pinecone
langchain-community
