import os
import tempfile
from typing import List, Dict, TypedDict

# LangChain v0.3+ imports
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

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
# Text Splitter
# -------------------------
def split_text(docs: List[Document], chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


# -------------------------
# Vector DB — FAISS (Cloud-safe)
# -------------------------
def create_vector_db(
    docs: List[Document],
    embeddings_model_name="sentence-transformers/all-MiniLM-L6-v2"
):
    if not docs:
        raise ValueError("No documents provided to create_vector_db.")

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    # Build an in-memory FAISS index
    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore


# -------------------------
# Graph State
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
    question = state.get("question", "")
    docs = vector_db.similarity_search(question, k=4)
    context = "\n\n".join([d.page_content for d in docs])
    return {"context": context}


def generate_response_node(state: GraphState) -> Dict:
    question = state.get("question", "")
    context = state.get("context", "")
    chat_history = state.get("chat_history", [])

    llm = ChatGroq(model_name=os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"))

    messages = [
        SystemMessage(
            content=(
                "You are a friendly and knowledgeable Fitness Assistant. "
                "Answer questions strictly based on the provided context. "
                "If the context doesn't contain the answer, say so clearly."
            )
        )
    ]

    # Add chat history
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    # Add current input
    messages.append(
        HumanMessage(
            content=f"Context:\n{context}\n\nQuestion:\n{question}"
        )
    )

    try:
        llm_response = llm.invoke(messages)
        text = llm_response.content
    except Exception as e:
        text = f"Error generating response: {e}"

    return {"response": text}


# -------------------------
# Build LangGraph Workflow
# -------------------------
def create_conversational_rag_chain(vector_db):
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve_context", lambda state: retrieve_context_node(state, vector_db))
    workflow.add_node("generate_response", generate_response_node)

    workflow.set_entry_point("retrieve_context")
    workflow.add_edge("retrieve_context", "generate_response")
    workflow.add_edge("generate_response", END)

    return workflow.compile()
