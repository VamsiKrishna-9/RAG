
import os
import sqlite3

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, "conversations.db")
MEMORY_PATH = os.path.join(BASE_DIR, "memory_index")


def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def initialize_memory_tracking():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS indexed_messages (
                message_id INTEGER PRIMARY KEY
            )
        """)


def index_pending_messages(embeddings):
    """
    Index messages that have not yet been added to memory FAISS.
    """

    initialize_memory_tracking()

    with get_db() as conn:

        rows = conn.execute("""
            SELECT
                m.id,
                m.conversation_id,
                m.role,
                m.content,
                c.title
            FROM messages m
            JOIN conversations c
                ON c.id = m.conversation_id
            LEFT JOIN indexed_messages i
                ON i.message_id = m.id
            WHERE i.message_id IS NULL
            ORDER BY m.id ASC
        """).fetchall()

    if not rows:
        return

    documents = []

    for row in rows:

        content = row["content"].strip()

        if not content:
            continue

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "message_id": row["id"],
                    "conversation_id": row["conversation_id"],
                    "role": row["role"],
                    "title": row["title"]
                }
            )
        )

    if not documents:
        return

    faiss_file = os.path.join(MEMORY_PATH, "index.faiss")
    pickle_file = os.path.join(MEMORY_PATH, "index.pkl")

    os.makedirs(MEMORY_PATH, exist_ok=True)

    if os.path.isfile(faiss_file) and os.path.isfile(pickle_file):

        memory_db = FAISS.load_local(
            MEMORY_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        memory_db.add_documents(documents)

    else:

        memory_db = FAISS.from_documents(
            documents,
            embeddings
        )

    # Persist FAISS before marking records as indexed.
    memory_db.save_local(MEMORY_PATH)

    with get_db() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO indexed_messages (message_id) VALUES (?)",
            [(doc.metadata["message_id"],) for doc in documents]
        )


def retrieve_memories(
    question,
    active_chat_id,
    embeddings,
    k=5
):
    """
    Retrieve relevant messages from previous conversations.
    """

    faiss_file = os.path.join(MEMORY_PATH, "index.faiss")
    pickle_file = os.path.join(MEMORY_PATH, "index.pkl")

    if not (
        os.path.isfile(faiss_file)
        and os.path.isfile(pickle_file)
    ):
        return ""

    memory_db = FAISS.load_local(
        MEMORY_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Retrieve extra results because the active conversation
    # will be excluded below.
    results = memory_db.similarity_search(
        question,
        k=max(k * 5, 20)
    )

    selected = []

    for document in results:

        metadata = document.metadata

        # Never retrieve the active chat as cross-chat memory.
        if metadata.get("conversation_id") == active_chat_id:
            continue

        selected.append(
            f"""
Previous conversation: {metadata.get('title', 'Untitled')}
Speaker: {metadata.get('role', 'unknown')}
Message: {document.page_content}
"""
        )

        if len(selected) >= k:
            break

    return "\n".join(selected)
