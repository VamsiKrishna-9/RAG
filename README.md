# MediAI – Retrieval-Augmented Generation (RAG) Chatbot

An intelligent, document-grounded chatbot built using **LangChain, FAISS, Hugging Face Embeddings, Groq LLM and Streamlit**. The chatbot retrieves relevant information from uploaded hospital knowledge-base documents and generates answers based on the retrieved context.

## Project Overview

The MediAI RAG chatbot is designed to answer questions using information available in hospital documents.

Instead of relying entirely on the language model's pre-trained knowledge, the application retrieves relevant document sections and provides them as context to the LLM.

If the requested information is not available in the knowledge base, the chatbot responds:

> I don't know based on the available MediAI documents.

### Key Features

- Document-based question answering.
- Retrieval-Augmented Generation using LangChain.
- PDF and Word document ingestion (depending on configured loaders).
- Text splitting with configurable chunk size and overlap.
- Semantic search using FAISS.
- Hugging Face sentence-transformer embeddings.
- LLM-powered response generation.
- Conversational follow-up question support.
- Streamlit-based interactive chat interface.
- Session-based chat history.
- Source-grounded answers to reduce hallucinations.

## Architecture

```text
              User
               |
               v
        Streamlit Interface
               |
               v
          User Question
               |
               v
       Question Rewriting
       (Conversational RAG)
               |
               v
         FAISS Retriever
               |
               v
       Relevant Document Chunks
               |
               v
          Context + Prompt
               |
               v
             LLM
               |
               v
        Generated Answer
               |
               v
        Streamlit Response
```

### Document Ingestion Pipeline

```text
Hospital Knowledge Base
          |
          v
     Document Loader
          |
          v
      Text Splitter
          |
          v
   Document Chunking
          |
          v
    Embedding Model
          |
          v
     Vector Creation
          |
          v
   FAISS Vector Store
```

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| LangChain | RAG application orchestration |
| FAISS | Vector storage and similarity search |
| Hugging Face | Text embedding models |
| Sentence Transformers | Semantic embeddings |
| Groq API | LLM inference |
| Streamlit | Interactive web interface |
| python-dotenv | Environment variable management |

## Project Structure

```text
project/
│
├── app.py
│
├── vectorstore/
│   ├── ingest.py
│   ├── .env.example
│   └── faiss_index/
│
├── data/
│   └── MediAI_Hospital_Knowledge_Base.docx
│
├── requirements.txt
├── .gitignore
└── README.md
```

**Note:** The FAISS index is generated locally during ingestion and is not required to be committed to GitHub. The knowledge-base document shown above is an example; only upload documents that you are authorized to share.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/VamsiKrishna-9/RAG.git
```

Navigate into the project:

```bash
cd RAG
```

### 2. Create a Virtual Environment

```bash
python -m venv rag
```

Activate it on Windows:

```bash
rag\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If you are using a Word document loader, ensure `docx2txt` is installed:

```bash
pip install docx2txt
```

### 4. Configure Environment Variables

Create a `.env` file inside the `vectorstore` folder.

Add:

```env
GROQ_API_KEY=your_groq_api_key
```

Never upload your actual `.env` file or API key to GitHub.

### 5. Prepare the Knowledge Base

Place your authorized PDF or Word documents inside the `data` folder.

Configure the appropriate document loader in `vectorstore/ingest.py`.

### 6. Generate the FAISS Vector Database

Navigate to the vectorstore folder:

```bash
cd vectorstore
```

Run:

```bash
python ingest.py
```

This process:

1. Loads the source documents.
2. Splits documents into smaller chunks.
3. Generates embeddings.
4. Creates the FAISS vector index.
5. Saves the index locally.

### 7. Run the Streamlit Application

Return to the project root:

```bash
cd ..
```

Start the application:

```bash
streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

## RAG Configuration

### Text Splitting

The current configuration uses:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
```

- `chunk_size=1000`: Approximately 1,000 characters per chunk.
- `chunk_overlap=200`: Approximately 200 characters shared between adjacent chunks.

This helps preserve context across document boundaries.

### Embedding Model

Current model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

This model converts document chunks and user questions into numerical vectors for semantic similarity search.

### FAISS Retriever

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
```

The retriever returns the four most similar document chunks for each query.

### LLM

The application uses a Groq-hosted language model to generate answers using the retrieved document context.

The model can be replaced with another supported LLM provider without redesigning the entire RAG architecture.

## Conversational RAG

The application supports follow-up questions using conversation history and question rewriting.

Example:

**User:** What is the MediAI leave policy?

**Chatbot:** Provides the leave policy based on retrieved documents.

**User:** Make it short.

**Chatbot:** Rewrites the follow-up question using the previous conversation and produces a shorter version of the same answer.

Conversation history is maintained in the current Streamlit session. Persistent conversation storage is not implemented.

## Hallucination Control

The chatbot follows a document-grounded answering approach:

- Use retrieved document context as the factual source.
- Do not invent hospital policies or operational details.
- Do not assume information that is absent from the knowledge base.
- Return an explicit fallback response when the required information cannot be found.
- Use conversation history to understand follow-up questions, not as an independent source of hospital facts.

## Future Enhancements

- Support multiple document formats.
- Improve retrieval with reranking.
- Add metadata filtering.
- Evaluate retrieval precision and answer faithfulness.
- Integrate live hospital data through authorized backend APIs.
- Implement role-based access to patient-specific information.
- Add multilingual question answering.
- Explore locally hosted LLMs and embedding models.
- Deploy the chatbot for secure application integration.

## Important Security Considerations

- Never commit API keys or credentials.
- Do not upload identifiable patient records or confidential hospital documents.
- Apply authentication and authorization before exposing hospital information.
- Use approved data-handling arrangements for external LLM providers.
- Protect sensitive information in logs and application responses.

## Author

**Vamsi Krishna Vadlamudi**

- GitHub: https://github.com/VamsiKrishna-9
- LinkedIn: https://www.linkedin.com/in/vamsi-krishna-vadlamudi-b1baa6263

---

*Built as a practical exploration of Retrieval-Augmented Generation, semantic search, LangChain and document-grounded conversational AI.*