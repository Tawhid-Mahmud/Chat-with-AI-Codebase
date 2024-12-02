"""
AI Chatbot for Codebase Analysis
--------------------------------
This application provides a chat interface to interact with and query codebases using RAG
(Retrieval Augmented Generation) and LLM technology.
"""

# Standard library imports
import os
import tempfile

# Third-party imports
import streamlit as st
from git import Repo
from groq import Groq
from github import Github
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

#######################
# Constants & Configs #
#######################

# Supported file extensions for code analysis
SUPPORTED_EXTENSIONS = {
    '.py', '.js', '.tsx', '.jsx', '.ipynb', 
    '.java', '.cpp', '.ts', '.go', '.rs', 
    '.vue', '.swift', '.c', '.h'
}

# Directories to ignore during processing
IGNORED_DIRS = {
    'node_modules', 'venv', 'env', 'dist', 
    'build', '.git', '__pycache__', '.next', 
    '.vscode', 'vendor'
}

# Available Groq models
GROQ_MODELS = {
    "llama-3.1-70b-versatile": "High performance, versatile model",
    "llama-3.1-8b-instant": "Fast, lightweight model",
    "mixtral-8x7b-32768": "Balanced performance model"
}

# System prompt for the AI assistant
SYSTEM_PROMPT = """
You are a technical expert. Provide detailed, accurate responses with code examples when appropriate. 
Focus on best practices and explain complex concepts clearly.
- Write the answer in 5 sentences or less
- Add code examples when appropriate
- If you don't know the answer, say so and don't make up an answer
- If you are not sure about the answer, clearly indicate your uncertainty
- If you can't answer, suggest the user ask a different question

ADD this to the answer: I am a technical assistant.
"""

######################
# Utility Functions #
######################

def clone_repository(repo_url):
    """
    Clone a GitHub repository to a temporary directory.
    
    Args:
        repo_url (str): URL of the GitHub repository
        
    Returns:
        str: Path to the cloned repository or None if failed
    """
    temp_dir = tempfile.mkdtemp()
    repo_name = repo_url.split("/")[-1]
    repo_path = os.path.join(temp_dir, repo_name)
    
    try:
        Repo.clone_from(repo_url, repo_path)
        return repo_path
    except Exception as e:
        st.error(f"Error cloning repository: {str(e)}")
        return None

def get_file_content(file_path, repo_path):
    """
    Read and return the content of a file.
    
    Args:
        file_path (str): Path to the file
        repo_path (str): Base path of the repository
        
    Returns:
        dict: File name and content or None if failed
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        rel_path = os.path.relpath(file_path, repo_path)
        return {"name": rel_path, "content": content}
    except Exception as e:
        st.error(f"Error processing file {file_path}: {str(e)}")
        return None

def get_main_files_content(repo_path):
    """
    Get content of all supported files in the repository.
    
    Args:
        repo_path (str): Path to the repository
        
    Returns:
        list: List of dictionaries containing file contents
    """
    files_content = []
    for root, _, files in os.walk(repo_path):
        if any(ignored_dir in root for ignored_dir in IGNORED_DIRS):
            continue
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.splitext(file)[1] in SUPPORTED_EXTENSIONS:
                file_content = get_file_content(file_path, repo_path)
                if file_content:
                    files_content.append(file_content)
    return files_content

# Initialize Pinecone
pc = Pinecone()
pinecone_index = pc.Index(name='codebase-rag')

# Function to get Hugging Face embeddings
def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(text)

# Function to get code chunks
def get_code_chunks(file_content):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_text(file_content)

# Function to perform RAG
def perform_rag(query, namespace):
    raw_query_embedding = get_huggingface_embeddings(query)
    top_matches = pinecone_index.query(vector=raw_query_embedding.tolist(), top_k=5, include_metadata=True, namespace=namespace)
    contexts = [item['metadata']['text'] for item in top_matches['matches']]
    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[:10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query
    # For a technical assistant
    llm_response = client.chat.completions.create(
        model=st.session_state["groq_model"],  # Use selected model from session state
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": augmented_query}
        ]
    )
    return llm_response.choices[0].message.content

#########################

# Initialize the Groq client with the API key
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Repository URL input
repo_url = st.text_input("Enter GitHub Repository URL:")

if repo_url:
    if 'repo_processed' not in st.session_state:
        st.session_state.repo_processed = False
    if not st.session_state.repo_processed:
        with st.spinner("Processing repository..."):
            # Clone repository
            repo_path = clone_repository(repo_url)
            
            # Get file contents
            file_content = get_main_files_content(repo_path)
            
            # Process and store embeddings
            documents = []
            for file in file_content:
                code_chunks = get_code_chunks(file['content'])
                for i, chunk in enumerate(code_chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={"source": file['name'], "chunk_id": i, "text": chunk}
                    )
                    documents.append(doc)
            
            vectorstore = PineconeVectorStore.from_documents(
                documents=documents,
                embedding=HuggingFaceEmbeddings(),
                index_name="codebase-rag",
                namespace=repo_url
            )
            
            st.session_state.repo_processed = True
            st.success("Repository processed and embeddings stored!")

GroqModel_1 = "llama-3.1-70b-versatile"
GroqModel_2 = "llama-3.1-8b-instant"
GroqModel_3 = "mixtral-8x7b-32768"

# Model selection options
model_options = [GroqModel_1, GroqModel_2, GroqModel_3]  

if "groq_model" not in st.session_state:
    st.session_state["groq_model"] = model_options[0]

# Initialize session state variables
if 'repo_processed' not in st.session_state:
    st.session_state.repo_processed = False

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Model selection dropdown
selected_model = st.selectbox("Select AI Model:", model_options, index=model_options.index(st.session_state["groq_model"]))
st.session_state["groq_model"] = selected_model

st.title("Ask your codebase")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
prompt = st.chat_input("Ask me something about the codebase")
if prompt:
    # Check if repository is processed
    if not st.session_state.repo_processed:
        st.warning("PLEASE ENTER A GITHUB REPOSITORY URL")
    else:
        # Display user message first
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Get RAG response
        response = perform_rag(prompt, repo_url)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Add messages to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response})