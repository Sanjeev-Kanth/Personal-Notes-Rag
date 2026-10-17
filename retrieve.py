from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

query = "What is encapsulation?"

results = vector_store.similarity_search(
    query,
    k=3
)

print("\nQuestion:")
print(query)

print("\nRetrieved chunks:")

for i, document in enumerate(results):
    print("\n" + "=" * 60)
    print(f"RESULT {i + 1}")
    print("=" * 60)

    print(document.page_content)
    print("\nMetadata:")
    print(document.metadata)