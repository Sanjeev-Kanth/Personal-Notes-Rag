from dotenv import load_dotenv
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# -----------------------------
# 1. Embedding model
# -----------------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

# -----------------------------
# 2. Connect to ChromaDB
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

# -----------------------------
# 4. Ask a question
# -----------------------------
question = "What is inheritance?"

# Retrieve relevant chunks
results = vector_store.similarity_search(
    question,
    k=3
)

# Combine retrieved chunks
context = "\n\n".join(
    document.page_content
    for document in results
)

# -----------------------------
# 5. RAG Prompt
# -----------------------------
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

# -----------------------------
# 6. Generate answer
# -----------------------------
response = llm.invoke(prompt)

# Handle Gemini response format
if isinstance(response.content, list):
    answer = response.content[0]["text"]
else:
    answer = response.content

print("\nQuestion:")
print(question)

print("\nAnswer:")
print(answer)