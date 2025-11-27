import streamlit as st
from dotenv import load_dotenv
from utils import (
    load_pdf,
    split_text,
    create_vector_db,
    create_conversational_rag_chain,
)
import os

load_dotenv()

st.set_page_config(page_title="Fitness Assistant AI", layout="wide")
st.title("Fitness Assistant AI (Powered by Groq Llama 3)")

# Session state
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# Check API Key
api_key_is_set = False
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    os.environ["GROQ_API_KEY"] = groq_api_key
    api_key_is_set = True
except Exception:
    if os.environ.get("GROQ_API_KEY"):
        api_key_is_set = True


# Sidebar
with st.sidebar:
    st.header("Setup")
    if api_key_is_set:
        st.success("Groq API Key is configured.")
    else:
        st.error("Add GROQ_API_KEY in Streamlit secrets or .env")

    uploaded_file = st.file_uploader("Upload a fitness/medical PDF", type=["pdf"])

    if st.button("Process Document", disabled=not api_key_is_set or not uploaded_file):
        with st.spinner("Processing PDF..."):
            try:
                docs = load_pdf(uploaded_file)
                chunks = split_text(docs)

                st.session_state.vector_db = create_vector_db(chunks)
                st.session_state.rag_chain = create_conversational_rag_chain(
                    st.session_state.vector_db
                )

                st.session_state.chat_history = []
                st.success("Document processed! Start chatting.")
            except Exception as e:
                st.error(f"Failed during processing: {e}")
                raise


# Chat interface
st.header("Chat with your AI Fitness Assistant")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.rag_chain:
    if prompt := st.chat_input("Ask anything about your fitness PDF"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
            graph_input = {
                "question": prompt,
                "chat_history": st.session_state.chat_history
            }

            result = st.session_state.rag_chain.invoke(graph_input)

            response = result.get("response", "")

            with st.chat_message("assistant"):
                st.markdown(response)

            st.session_state.chat_history.append(
                {"role": "assistant", "content": response}
            )

else:
    if not api_key_is_set:
        st.warning("Configure Groq API key first.")
    else:
        st.warning("Upload a PDF and click 'Process Document'.")
