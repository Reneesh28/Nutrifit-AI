import os
import tempfile
from typing import List, Dict, TypedDict

# Updated imports for LangChain v0.3+
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import LanceDB
from langchain_huggingface import HuggingFaceEmbeddings

import lancedb
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq


# -------------------------
# PDF Loader
# -------------------------
def load_pdf(uploaded_file) -> List[Document]:
    """
    Loads a PDF file from Streamlit uploader and converts it into LangChain Document objects.
    """
    if uploaded_file is None:
        return []

    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
    finally:
        # Clean up the temporary file
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return documents


# -------------------------
# Text Splitter
# -------------------------
def split_text(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Splits documents into manageable chunks for vector embeddings.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


# -------------------------
# Vector Database (LanceDB)
# -------------------------
def create_vector_db(
    docs: List[Document],
    embeddings_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    table_name: str = "fitness_capstone",
):
    """
    Creates or loads a LanceDB vector database using the given documents.
    """
    if not docs:
        raise ValueError("No documents provided to create_vector_db.")

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    # Ensure LanceDB directory exists
    lance_path = os.environ.get("LANCEDB_PATH", "/tmp/lancedb")
    os.makedirs(lance_path, exist_ok=True)

    conn = lancedb.connect(lance_path)
    db = LanceDB.from_documents(docs, embeddings, connection=conn, table_name=table_name)

    return db


# -------------------------
# Graph State Definition
# -------------------------
class GraphState(TypedDict):
    question: str
    context: str
    chat_history: List[Dict]
    response: str


# -------------------------
# RAG Nodes
# -------------------------
def retrieve_context_node(state: GraphState, vector_db) -> Dict:
    """
    Retrieves the most relevant text chunks for the given user query from LanceDB.
    """
    question = state.get("question", "")
    docs = vector_db.similarity_search(question, k=4)
    context = "\n\n".join([getattr(d, "page_content", "") for d in docs])

    return {"context": context}


def generate_response_node(state: GraphState) -> Dict:
    """
    Uses Groq LLM to generate a response using the retrieved context.
    """
    question = state.get("question", "")
    context = state.get("context", "")
    chat_history = state.get("chat_history", [])

    llm = ChatGroq(model_name=os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"))

    # System prompt
    messages = [
        SystemMessage(
            content=(
                "You are a friendly and knowledgeable Fitness Assistant. "
                "Answer questions strictly based on the provided context. "
                "If the context doesn't contain the answer, say so clearly."
            )
        )
    ]

    # Add previous chat history
    for msg in chat_history:
        role, content = msg.get("role"), msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    # Add the current user question and context
    user_message = f"Context:\n{context}\n\nQuestion:\n{question}"
    messages.append(HumanMessage(content=user_message))

    # Generate response
    try:
        llm_response = llm.invoke(messages)
        text = getattr(llm_response, "content", None) or getattr(llm_response, "text", None) or str(llm_response)
    except Exception as e:
        text = f"Error generating response: {e}"

    return {"response": text}


# -------------------------
# Build LangGraph Workflow
# -------------------------
def create_conversational_rag_chain(vector_db):
    """
    Creates a LangGraph workflow with two steps:
      1. Retrieve context from LanceDB
      2. Generate response using Groq LLM
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve_context", lambda state: retrieve_context_node(state, vector_db))
    workflow.add_node("generate_response", generate_response_node)

    workflow.set_entry_point("retrieve_context")
    workflow.add_edge("retrieve_context", "generate_response")
    workflow.add_edge("generate_response", END)

    return workflow.compile()
