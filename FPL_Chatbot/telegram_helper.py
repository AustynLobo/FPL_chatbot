"""
telegram_helper.py
==================
All Telegram API send functions.
Imported by lambda_function.py.
"""

import json
import os
import urllib.request

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_API   = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"


def send_message(chat_id, text):
    """Send a plain text message to a Telegram chat."""
    payload = json.dumps({
        "chat_id"   : chat_id,
        "text"      : text,
        "parse_mode": "Markdown"
    }).encode()

    req = urllib.request.Request(
        f"{TELEGRAM_API}/sendMessage",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.loads(response.read())


def send_photo(chat_id, photo_url, caption=""):
    """Send a photo to a Telegram chat using a URL or presigned S3 URL."""
    payload = json.dumps({
        "chat_id"   : chat_id,
        "photo"     : photo_url,
        "caption"   : caption,
        "parse_mode": "Markdown"
    }).encode()

    req = urllib.request.Request(
        f"{TELEGRAM_API}/sendPhoto",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=15) as response:
        return json.loads(response.read())


def send_typing_action(chat_id):
    """Show the 'typing...' indicator while the bot is processing."""
    payload = json.dumps({
        "chat_id": chat_id,
        "action" : "typing"
    }).encode()

    req = urllib.request.Request(
        f"{TELEGRAM_API}/sendChatAction",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    urllib.request.urlopen(req, timeout=5)