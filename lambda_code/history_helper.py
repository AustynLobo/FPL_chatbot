"""
history_helper.py
==================
DynamoDB conversation history read/write.
Imported by lambda_function.py.
"""

import os
import boto3

dynamodb    = boto3.resource("dynamodb")
table       = dynamodb.Table(os.environ["DYNAMODB_TABLE"])
MAX_HISTORY = 10   # keep last 10 messages per user


def get_history(chat_id):
    """Load conversation history for a chat from DynamoDB."""
    try:
        response = table.get_item(Key={"chat_id": str(chat_id)})
        return response.get("Item", {}).get("messages", [])
    except Exception:
        return []


def save_history(chat_id, messages):
    """Persist conversation history to DynamoDB, capped at MAX_HISTORY messages."""
    try:
        messages = messages[-MAX_HISTORY:]
        table.put_item(Item={
            "chat_id" : str(chat_id),
            "messages": messages
        })
    except Exception:
        pass


def clear_history(chat_id):
    """Wipe conversation history for a chat (called on /start)."""
    save_history(chat_id, [])