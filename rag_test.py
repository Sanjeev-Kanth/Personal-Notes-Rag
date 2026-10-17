from dotenv import load_dotenv
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_chroma import Chroma

load_dotenv()

# 1. Load embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

# 2. Connect to existing Chroma database
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 3. User question
question = "What is encapsulation?"

# 4. Retrieve relevant chunks
results = vector_store.similarity_search(
    question,
    k=3
)

# 5. Combine retrieved chunks into context
context = "\n\n".join(
    document.page_content
    for document in results
)

print("\nRetrieved Context:")
print(context)

# 6. Create Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0
)

# 7. Create RAG prompt
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

# 8. Send prompt to Gemini
response = llm.invoke(prompt)

# 9. Display answer
print("\nQuestion:")
print(question)

if isinstance(response.content, list):
    answer = response.content[0]["text"]
else:
    answer = response.content

print("\nAnswer:")
print(answer)