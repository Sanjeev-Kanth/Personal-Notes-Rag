from dotenv import load_dotenv
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_chroma import Chroma

load_dotenv()

# -----------------------------
# 1. Embedding model
# -----------------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

# -----------------------------
# 2. Chroma vector database
# -----------------------------
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# -----------------------------
# 3. Gemini LLM
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0
)

print("=" * 60)
print("       PERSONAL NOTES RAG CHATBOT")
print("=" * 60)
print("Ask questions about your notes.")
print("Type 'exit' to quit.")
print("=" * 60)

# -----------------------------
# 4. Chat loop
# -----------------------------
while True:

    question = input("\nYou: ")

    # Exit condition
    if question.lower() == "exit":
        print("\nGoodbye!")
        break

    # Ignore empty input
    if not question.strip():
        continue

    # -------------------------
    # Retrieve relevant chunks
    # -------------------------
    results = vector_store.similarity_search(
        question,
        k=3
    )

    # -------------------------
    # Build context
    # -------------------------
    context = "\n\n".join(
        document.page_content
        for document in results
    )

    # -------------------------
    # RAG prompt
    # -------------------------
    prompt = f"""
You are a helpful assistant answering questions
using the user's personal notes.

Use ONLY the context provided below.

If the answer cannot be found in the context,
say "I don't know based on your notes."

Context:
{context}

Question:
{question}

Answer:
"""

    # -------------------------
    # Generate response
    # -------------------------
    response = llm.invoke(prompt)

    # Handle Gemini response format
    if isinstance(response.content, list):
        answer = response.content[0]["text"]
    else:
        answer = response.content

    print("\nBot:", answer)