"""
lambda_function.py
==================
AWS Lambda entry point. Thin router only — no business logic lives here.

Request routing order:
  1. /start command         → reset history + welcome message
  2. TOTW image request     → presigned S3 URL + Telegram sendPhoto
  3. FPL text question      → Claude API + Telegram sendMessage
  4. Non-FPL message        → polite rejection

All logic lives in the helper modules:
  telegram_helper.py     — Telegram send functions
  history_helper.py      — DynamoDB conversation history
  predictions_helper.py  — S3 predictions CSV + Claude API
  totw_helper.py         — TOTW detection + S3 presigned URL + photo send
"""

import json
import logging
import urllib.error

from telegram_helper    import send_message, send_typing_action
from history_helper     import get_history, save_history, clear_history
from predictions_helper import is_fpl_related, get_latest_predictions, ask_claude
from totw_helper        import is_totw_request, handle_totw_request

logger = logging.getLogger()
logger.setLevel(logging.INFO)

WELCOME_MESSAGE = (
    "👋 Welcome to the *FPL Predictor Bot*!\n\n"
    "Ask me anything about this gameweek:\n"
    "• Who are the best value midfielders?\n"
    "• Which defenders have easy fixtures?\n"
    "• Who should I captain this week?\n\n"
    "📊 *Team of the Week:*\n"
    "• Show me the actual TOTW\n"
    "• Show me the predicted TOTW"
)

NOT_FPL_MESSAGE = (
    "I only answer FPL related questions! Try asking:\n"
    "• Who should I captain this week?\n"
    "• Best value midfielders?\n"
    "• Which defenders have easy fixtures?\n"
    "• Show me the actual TOTW\n"
    "• Show me the predicted TOTW"
)


def lambda_handler(event, context):
    try:
        body         = json.loads(event.get("body", "{}"))
        message      = body.get("message", {})
        chat_id      = message.get("chat", {}).get("id")
        user_message = message.get("text", "")

        logger.info(f"Incoming Event: {json.dumps(event)}")

        if not chat_id or not user_message:
            return {"statusCode": 200, "body": "ok"}

        # ── 1. /start ─────────────────────────────────────────────────────────
        if user_message == "/start":
            clear_history(chat_id)
            send_message(chat_id, WELCOME_MESSAGE)
            return {"statusCode": 200, "body": "ok"}

        # ── 2. TOTW image request ─────────────────────────────────────────────
        if is_totw_request(user_message):
            send_typing_action(chat_id)
            handle_totw_request(chat_id, user_message)
            return {"statusCode": 200, "body": "ok"}

        # ── 3. Non-FPL message ────────────────────────────────────────────────
        if not is_fpl_related(user_message):
            send_message(chat_id, NOT_FPL_MESSAGE)
            return {"statusCode": 200, "body": "ok"}

        # ── 4. FPL text question → Claude ─────────────────────────────────────
        send_typing_action(chat_id)

        history     = get_history(chat_id)
        predictions = get_latest_predictions()
        answer      = ask_claude(user_message, predictions, history)

        history.append({"role": "user",      "content": user_message})
        history.append({"role": "assistant", "content": answer})
        save_history(chat_id, history)

        send_message(chat_id, answer)
        return {"statusCode": 200, "body": "ok"}

    except urllib.error.HTTPError as e:
        error_body = e.read().decode()
        logger.error(f"HTTP error: {e.code} — {error_body}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"HTTP {e.code}", "detail": error_body})
        }
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }