"""
predictions_helper.py
======================
Loads the latest FPL predictions CSV from S3 and calls the Claude API.
Imported by lambda_function.py.
"""

import csv
import io
import json
import os
import urllib.request

import boto3

S3_BUCKET         = os.environ["S3_BUCKET"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

FPL_KEYWORDS = [
    "player", "midfielder", "defender", "forward", "goalkeeper",
    "captain", "transfer", "price", "fixture", "fdr", "points",
    "gw", "gameweek", "buy", "sell", "bench", "squad", "team",
    "best", "value", "cheap", "recommend", "who", "should",
    "vice captain", "triple captain",
]


def is_fpl_related(message):
    """Return True if the message contains at least one FPL keyword."""
    msg = message.lower()
    return any(keyword in msg for keyword in FPL_KEYWORDS)


def get_latest_predictions():
    """
    Fetch the latest fpl_best_by_position CSV from S3 and format it
    as a plain-text string for use as Claude context.
    """
    s3 = boto3.client("s3")

    response = s3.list_objects_v2(
        Bucket=S3_BUCKET,
        Prefix="predictions/fpl_best_by_position_"
    )
    files  = [obj["Key"] for obj in response.get("Contents", [])]
    latest = sorted(files)[-1]

    obj     = s3.get_object(Bucket=S3_BUCKET, Key=latest)
    content = obj["Body"].read().decode("utf-8")

    reader = csv.DictReader(io.StringIO(content))
    rows   = list(reader)

    lines       = []
    current_pos = None
    for row in rows:
        if row["Pos"] != current_pos:
            current_pos = row["Pos"]
            lines.append(f"\n{current_pos}:")
        lines.append(
            f"  {row['Player']:<20} Price: £{row['Price(£m)']}  "
            f"PredPts: {row['PredPts']}  FDR: {row['FDR']}  "
            f"Home: {row['Home']}  Value: {row['Value']}"
        )

    gw = latest.split("gw")[1].replace(".csv", "")
    return f"GW{gw} Predictions:\n" + "\n".join(lines)


def ask_claude(user_message, predictions_context, history):
    """
    Send the user message + predictions context + conversation history
    to the Claude API and return the response text.
    """
    messages = history + [
        {
            "role"   : "user",
            "content": f"FPL data:\n{predictions_context}\n\nQuestion: {user_message}"
        }
    ]

    payload = json.dumps({
        "model"     : "claude-haiku-4-5-20251001",
        "max_tokens": 1024,
        "system"    : (
            "You are an FPL (Fantasy Premier League) assistant. "
            "Answer questions using the prediction data provided. "
            "Be concise and helpful. Always mention player prices "
            "and predicted points when recommending players. "
            "Keep responses under 200 words as this is a Telegram chat."
        ),
        "messages": messages
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type"     : "application/json",
            "x-api-key"        : ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01"
        }
    )

    with urllib.request.urlopen(req, timeout=30) as response:
        result = json.loads(response.read())
        return result["content"][0]["text"]