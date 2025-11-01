import os
import tempfile
from typing import List, Dict, TypedDict

from langchain.schema import Document, SystemMessage, HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import LanceDB
import lancedb

from langgraph.graph import END, StateGraph
from langchain_groq import ChatGroq


# -------------------------
# PDF loader
# -------------------------
def load_pdf(uploaded_file) -> List[Document]:
    """
    Loads a PDF file and returns a list of LangChain Document objects.
    Works with file-like objects (e.g., Streamlit UploadedFile or local files).
    """
    if uploaded_file is None:
        return []

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return documents


# -------------------------
# Text splitter
# -------------------------
def split_text(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Splits LangChain Document objects into smaller chunks for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


# -------------------------
# Create LanceDB vector DB
# -------------------------
def create_vector_db(
    docs: List[Document],
    embeddings_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    table_name: str = "fitness_capstone",
):
    """
    Create (or open) a LanceDB-backed LangChain vectorstore from documents.
    """
    if not docs:
        raise ValueError("No documents provided to create_vector_db.")

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    lance_path = os.environ.get("LANCEDB_PATH", "/tmp/lancedb")
    os.makedirs(lance_path, exist_ok=True)

    conn = lancedb.connect(lance_path)
    db = LanceDB.from_documents(docs, embeddings, connection=conn, table_name=table_name)
    return db


# -------------------------
# LangGraph + Groq LLM RAG chain
# -------------------------
class GraphState(TypedDict):
    question: str
    context: str
    chat_history: List[Dict]
    response: str


def retrieve_context_node(state: GraphState, vector_db) -> Dict:
    """
    Node that retrieves relevant passages from the vectorstore.
    """
    question = state.get("question", "")
    docs = vector_db.similarity_search(question, k=4)
    context = "\n\n".join([getattr(d, "page_content", "") for d in docs])
    return {"context": context}


def generate_response_node(state: GraphState) -> Dict:
    """
    Node that calls Groq Chat LLM to generate an answer given question + context.
    """
    question = state.get("question", "")
    context = state.get("context", "")
    chat_history = state.get("chat_history", [])

    llm = ChatGroq(model_name=os.environ.get("GROQ_MODEL", "llama-3.1-70b-versatile"))

    # Build message chain using proper LangChain message objects
    messages = [SystemMessage(content=(
        "You are a friendly and encouraging Fitness Assistant. "
        "Provide accurate answers based ONLY on the provided context. "
        "If the answer is not in the context, say you cannot answer based on the document."
    ))]

    # Convert chat_history (list of dicts) to proper message objects
    for msg in chat_history:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    # Append user message for this turn
    user_content = f"Context:\n{context}\n\nQuestion:\n{question}"
    messages.append(HumanMessage(content=user_content))

    # Call Groq LLM safely
    try:
        llm_response = llm.invoke(messages)
        text = getattr(llm_response, "content", None) or getattr(llm_response, "text", None) or str(llm_response)
    except Exception:
        try:
            llm_response = llm(messages)
            if isinstance(llm_response, str):
                text = llm_response
            else:
                text = getattr(llm_response, "content", None) or str(llm_response)
        except Exception as e:
            text = f"Error generating response: {e}"

    return {"response": text}


def create_conversational_rag_chain(vector_db):
    """
    Creates a LangGraph StateGraph with two nodes:
      1) retrieve_context (vector search)
      2) generate_response (Groq LLM)
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve_context", lambda state: retrieve_context_node(state, vector_db))
    workflow.add_node("generate_response", generate_response_node)

    workflow.set_entry_point("retrieve_context")
    workflow.add_edge("retrieve_context", "generate_response")
    workflow.add_edge("generate_response", END)

    return workflow.compile()
