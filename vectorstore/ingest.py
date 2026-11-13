
import os
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Document path
DATA_PATH = "../data/MediAI_Hospital_Knowledge_Base.docx"

# Vector database path
VECTORSTORE_PATH = "faiss_index"

# 1. Load Word document
loader = Docx2txtLoader(DATA_PATH)
documents = loader.load()

print("Document loaded successfully!")
print("Total documents:", len(documents))

# 2. Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print("Total chunks:", len(chunks))

# 3. Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Create FAISS vector database
vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

# 5. Save vector database
vectorstore.save_local(VECTORSTORE_PATH)

print("MediAI Knowledge Base indexed successfully!")
