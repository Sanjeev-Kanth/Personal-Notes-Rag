import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_chroma import Chroma

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Personal Notes RAG",
    page_icon="📚",
    layout="centered"
)

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("📚 Personal Notes RAG Chatbot")
st.caption("Ask questions based on your personal notes.")

# --------------------------------------------------
# Load models and vector database
# --------------------------------------------------

@st.cache_resource
def load_rag_system():

    # Embedding model
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    # Existing Chroma database
    vector_store = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    # Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0
    )

    return vector_store, llm


vector_store, llm = load_rag_system()

# --------------------------------------------------
# Initialize chat history
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------------------------
# Display previous messages
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --------------------------------------------------
# Chat input
# --------------------------------------------------

question = st.chat_input("Ask something about your notes...")

if question:

    # Display user's question
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # --------------------------------------------------
    # Retrieve relevant documents
    # --------------------------------------------------

    results = vector_store.similarity_search(
        question,
        k=3
    )

    # --------------------------------------------------
    # Create context
    # --------------------------------------------------

    context = "\n\n".join(
        document.page_content
        for document in results
    )

    # --------------------------------------------------
    # RAG prompt
    # --------------------------------------------------

    prompt = f"""
You are a helpful assistant answering questions
using the user's personal notes.

Use ONLY the information provided in the context below.

If the answer cannot be found in the context,
say exactly:

"I don't know based on your notes."

Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""

    # --------------------------------------------------
    # Generate answer
    # --------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Searching your notes..."):

            response = llm.invoke(prompt)

            # Gemini may return structured content
            if isinstance(response.content, list):
                answer = response.content[0]["text"]
            else:
                answer = response.content

            st.markdown(answer)

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("📖 About")

    st.write(
        """
        This chatbot uses Retrieval-Augmented Generation (RAG)
        to answer questions from your personal notes.
        """
    )

    st.divider()

    st.subheader("Architecture")

    st.write(
        """
        📝 Notes  
        ↓  
        ✂️ Chunking  
        ↓  
        🔢 Embeddings  
        ↓  
        🗄️ ChromaDB  
        ↓  
        🔍 Similarity Search  
        ↓  
        🤖 Gemini  
        ↓  
        💬 Answer
        """
    )

    st.divider()

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()