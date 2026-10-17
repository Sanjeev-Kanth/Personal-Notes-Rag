import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

text = "Java supports inheritance."

vector = embeddings.embed_query(text)

print("Vector type:", type(vector))
print("Vector length:", len(vector))
print("First 10 values:", vector[:10])