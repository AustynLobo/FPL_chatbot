"""
totw_helper.py
==============
Team of the Week (TOTW) request detection, S3 presigned URL generation
and Telegram photo delivery.
Imported by lambda_function.py.
"""

import json
import logging
import os

import boto3

from telegram_helper import send_photo, send_message

logger    = logging.getLogger()
S3_BUCKET = os.environ["S3_BUCKET"]

# Phrases that trigger the TOTW image flow
TOTW_TRIGGERS = [
    "totw",
    "team of the week",
    "best 11",
    "best xi",
    "predicted 11",
    "predicted xi",
    "lineup",
    "show team",
    "show me the team",
]

# Words that indicate the user wants the predicted (upcoming) team
PREDICT_WORDS = ["predict", "predicted", "next", "upcoming"]


def is_totw_request(message):
    """Return True if the message is asking for a TOTW image."""
    msg = message.lower()
    return any(trigger in msg for trigger in TOTW_TRIGGERS)


def detect_totw_mode(message):
    """
    Return 'predict' if the user mentions next/upcoming/predicted,
    otherwise return 'actual'.
    """
    msg = message.lower()
    if any(word in msg for word in PREDICT_WORDS):
        return "predict"
    return "actual"


def get_current_gw():
    """
    Read last_finished_gw from cache/meta.json stored in S3.
    Falls back to 1 if the file is missing or unreadable.
    """
    try:
        s3  = boto3.client("s3")
        obj = s3.get_object(Bucket=S3_BUCKET, Key="cache/meta.json")
        data = json.loads(obj["Body"].read().decode("utf-8"))
        return int(data.get("last_finished_gw", 1))
    except Exception as e:
        logger.error(f"Error reading cache/meta.json from S3: {e}")
        return 1


def get_presigned_url(mode, gw):
    """
    Generate a 5-minute presigned URL for the TOTW PNG in S3.

    Key structure:
        totw/actual/fpl_totw_actual_gw{N}.png
        totw/predict/fpl_totw_predicted_gw{N}.png
    """
    s3          = boto3.client("s3")
    file_suffix = "predicted" if mode == "predict" else "actual"
    key         = f"totw/{mode}/fpl_totw_{file_suffix}_gw{gw}.png"

    try:
        # Confirm the object exists before signing
        s3.head_object(Bucket=S3_BUCKET, Key=key)

        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": key},
            ExpiresIn=300   # 5 minutes
        )
        logger.info(f"Presigned URL generated: {key}")
        return url

    except Exception as e:
        logger.warning(f"TOTW file not found in S3: {key} — {e}")
        return None


def handle_totw_request(chat_id, message):
    """
    Full TOTW request pipeline:
      1. Detect actual vs predicted mode
      2. Determine the target GW
      3. Generate presigned URL
      4. Send photo or fallback error message to Telegram

    Returns True if a photo was sent, False otherwise.
    """
    mode      = detect_totw_mode(message)
    gw        = get_current_gw()
    target_gw = gw + 1 if mode == "predict" else gw

    logger.info(f"TOTW request — mode={mode}, target_gw={target_gw}")

    image_url = get_presigned_url(mode, target_gw)

    if image_url:
        label   = "Predicted XI" if mode == "predict" else "Team of the Week"
        caption = (
            f"📊 *GW{target_gw} {label}*\n"
            f"_Formation auto-selected · max 3 players per club · £100m budget_"
        )
        send_photo(chat_id, image_url, caption)
        return True

    # File not in S3 — give the user a clear explanation
    mode_label = "predicted" if mode == "predict" else "actual"
    send_message(
        chat_id,
        f"⚠️ The {mode_label} TOTW image for GW{target_gw} isn't available yet.\n\n"
        f"It's generated after each gameweek completes. "
        f"Try again later or ask me a text question instead."
    )
    return False