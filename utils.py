import os
import tempfile
from typing import List, Dict, TypedDict

from langchain.schema import Document
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
    Loads a PDF file (works for both Streamlit UploadedFile and local file path).
    Returns a list of LangChain Document objects.
    """
    if uploaded_file is None:
        return []

    # Check if it's a Streamlit UploadedFile (has getvalue)
    if hasattr(uploaded_file, "getvalue"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
    else:
        # Assume it's a file path
        tmp_path = uploaded_file

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
    finally:
        # If we created a temp file (Streamlit case), remove it
        if hasattr(uploaded_file, "getvalue"):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    return documents


# -------------------------
# Text splitter
# -------------------------
def split_text(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


# -------------------------
# Create LanceDB vector DB
# -------------------------
def create_vector_db(docs: List[Document], embeddings_model_name: str = "sentence-transformers/all-MiniLM-L6-v2", table_name: str = "fitness_capstone"):
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
    question = state.get("question", "")
    docs = vector_db.similarity_search(question, k=4)
    context = "\n\n".join([getattr(d, "page_content", "") for d in docs])
    return {"context": context}


def generate_response_node(state: GraphState) -> Dict:
    question = state.get("question", "")
    context = state.get("context", "")
    chat_history = state.get("chat_history", [])

    llm = ChatGroq(model_name=os.environ.get("GROQ_MODEL", "llama3-70b-8192"))

    system_prompt = (
        "You are a friendly and encouraging Fitness Assistant. Provide accurate answers based ONLY on the provided context. "
        "If the answer is not in the context, say you cannot answer based on the document."
    )

    messages = [{"role": "system", "content": system_prompt}]
    if chat_history:
        messages.extend(chat_history)

    user_content = f"Context:\n{context}\n\nQuestion:\n{question}"
    messages.append({"role": "user", "content": user_content})

    try:
        llm_response = llm.invoke(messages)
        text = getattr(llm_response, "content", None) or getattr(llm_response, "text", None) or str(llm_response)
    except Exception:
        llm_response = llm(messages)
        text = llm_response if isinstance(llm_response, str) else getattr(llm_response, "content", None) or str(llm_response)

    return {"response": text}


def create_conversational_rag_chain(vector_db):
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve_context", lambda state: retrieve_context_node(state, vector_db))
    workflow.add_node("generate_response", generate_response_node)
    workflow.set_entry_point("retrieve_context")
    workflow.add_edge("retrieve_context", "generate_response")
    workflow.add_edge("generate_response", END)
    return workflow.compile()
