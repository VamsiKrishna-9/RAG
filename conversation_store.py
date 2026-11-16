
import sqlite3
import uuid
from datetime import datetime
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "conversations.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def initialize_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT DEFAULT '[]',
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(id)
                    ON DELETE CASCADE
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_conversation
            ON messages(conversation_id, id)
        """)


def create_conversation():
    chat_id = str(uuid.uuid4())
    now = datetime.now().isoformat()

    with get_connection() as conn:
        conn.execute("""
            INSERT INTO conversations
            (id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
        """, (chat_id, "New Chat", now, now))

    return chat_id


def get_all_conversations():
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM conversations
            ORDER BY updated_at DESC
        """).fetchall()

    return [dict(row) for row in rows]


def get_messages(chat_id):
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT role, content, sources, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
        """, (chat_id,)).fetchall()

    messages = []

    for row in rows:
        message = dict(row)
        message["sources"] = json.loads(
            message["sources"] or "[]"
        )
        messages.append(message)

    return messages


def save_message(chat_id, role, content, sources=None):
    now = datetime.now().isoformat()

    with get_connection() as conn:
        conversation = conn.execute("""
            SELECT title FROM conversations WHERE id = ?
        """, (chat_id,)).fetchone()

        if conversation is None:
            raise ValueError("Conversation not found.")

        conn.execute("""
            INSERT INTO messages
            (conversation_id, role, content, sources, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            chat_id,
            role,
            content,
            json.dumps(sources or [], ensure_ascii=False),
            now
        ))

        # Generate title from the first user message.
        if role == "user" and conversation["title"] == "New Chat":
            title = content.strip().replace("\n", " ")[:45]
            if len(content.strip()) > 45:
                title += "..."

            conn.execute("""
                UPDATE conversations
                SET title = ?, updated_at = ?
                WHERE id = ?
            """, (title or "New Chat", now, chat_id))

        else:
            conn.execute("""
                UPDATE conversations
                SET updated_at = ?
                WHERE id = ?
            """, (now, chat_id))


def delete_conversation(chat_id):
    with get_connection() as conn:
        conn.execute("""
            DELETE FROM conversations WHERE id = ?
        """, (chat_id,))
