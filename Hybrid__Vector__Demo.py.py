from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import OracleVS
from langchain_community.vectorstores.utils import DistanceStrategy
import oracledb
from datetime import datetime

# Step 1: Load text documents
loader = TextLoader("my_docs.txt")
documents = loader.load()

# Step 2: Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Step 3: Add metadata (for SQL filtering)
for d in docs:
    d.metadata = {
        "category": "LangChain",
        "created_on": datetime.now().strftime("%Y-%m-%d")
    }

# Step 4: Create embeddings using Ollama
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

# Step 5: Create Oracle connection pool
client = oracledb.create_pool(
    user="student",
    password="apelix",
    dsn="localhost:1521/FREEPDB1",
    min=1, max=5, increment=1
)

# Step 6: Create Oracle vector store table
vector_store = OracleVS.from_documents(
    documents=docs,
    embedding=embeddings,
    client=client,
    table_name="HYBRID_VSTORE",
    distance_strategy=DistanceStrategy.COSINE,
    metadata_columns={
        "category": "VARCHAR2(100)",
        "created_on": "DATE"
    }
)

# Step 7: Example hybrid query
query = "Explain LangGraph framework"
results = vector_store.similarity_search(
    query,
    k=3,
    where_clause="category = 'LangChain' AND created_on >= DATE '2024-01-01'"
)

# Step 8: Display results
print(f"\n Found {len(results)} hybrid matches:\n")
for i, r in enumerate(results, start=1):
    print(f"{i}. {r.page_content[:200]}...\nMetadata: {r.metadata}\n---")




