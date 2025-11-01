from langchain.schema import HumanMessage, SystemMessage, AIMessage

def generate_response_node(state: dict) -> dict:
    question = state.get("question", "")
    context = state.get("context", "")
    chat_history = state.get("chat_history", [])

    llm = ChatGroq(model_name=os.environ.get("GROQ_MODEL", "llama3-70b-8192"))

    # Build conversation messages (convert to LangChain message objects)
    messages = [SystemMessage(content=(
        "You are a friendly and encouraging Fitness Assistant. "
        "Provide accurate answers based ONLY on the provided context. "
        "If the answer is not in the context, say you cannot answer based on the document."
    ))]

    # Convert previous chat history to LangChain messages
    for m in chat_history:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    # Add current user message (question + context)
    user_content = f"Context:\n{context}\n\nQuestion:\n{question}"
    messages.append(HumanMessage(content=user_content))

    # Invoke LLM
    try:
        llm_response = llm.invoke(messages)
        text = getattr(llm_response, "content", None) or str(llm_response)
    except Exception as e:
        text = f"Error while generating response: {e}"

    return {"response": text}
