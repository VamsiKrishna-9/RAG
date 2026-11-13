import os

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# =========================================================
# 1. PROJECT PATHS
# This file is located at: RAG/vectorstore/ingest.py
# =========================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

DATA_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "MediAI_Hospital_Knowledge_Base.docx"
)

# Keep this path identical to the FAISS_PATH in app.py.
FAISS_PATH = os.path.join(
    SCRIPT_DIR,
    "faiss_index"
)


def main():
    # =====================================================
    # 2. CHECK AND LOAD WORD DOCUMENT
    # =====================================================

    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(
            "Knowledge-base document not found:\n"
            f"{DATA_PATH}\n"
            "Place MediAI_Hospital_Knowledge_Base.docx inside "
            "the project's data folder."
        )

    loader = Docx2txtLoader(DATA_PATH)
    documents = loader.load()

    if not documents:
        raise ValueError("The Word document contains no readable text.")

    print("Document loaded successfully!")
    print("Total documents:", len(documents))

    # =====================================================
    # 3. SPLIT DOCUMENT INTO CHUNKS
    # =====================================================

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No chunks were created from the document.")

    print("Total chunks:", len(chunks))

    # =====================================================
    # 4. CREATE EMBEDDINGS
    # Keep the same model used in app.py.
    # =====================================================

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # =====================================================
    # 5. CREATE FAISS VECTOR DATABASE
    # =====================================================

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # =====================================================
    # 6. SAVE FAISS INDEX
    # =====================================================

    os.makedirs(FAISS_PATH, exist_ok=True)
    vectorstore.save_local(FAISS_PATH)

    print("MediAI Knowledge Base indexed successfully!")
    print("FAISS index saved to:", FAISS_PATH)


if __name__ == "__main__":
    main()
