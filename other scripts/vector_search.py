from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path
import os

INDEX_DIR = "faiss_index"
docs_dir = Path("docs")  
all_docs = []
for file in docs_dir.glob("*.txt"):
    loader = TextLoader(str(file), encoding="utf-8")
    all_docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(all_docs)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)
vectorstore = FAISS.from_documents(chunks, embeddings)

if os.path.exists(INDEX_DIR):
    print("Loading existing FAISS index...")
    vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
else:
    print("Building new FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_DIR)

query = "What is machine learning?"
results = vectorstore.similarity_search(query, k=3)  

print("\n=== Rezultate căutare ===")
for i, doc in enumerate(results, 1):
    print(f"[{i}] {doc.page_content[:200]}...\n")
