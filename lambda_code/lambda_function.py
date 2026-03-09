import json
import boto3
import csv
import io
import urllib.request
import urllib.parse
import os

S3_BUCKET = "my-fpl-predictions"
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"


def get_latest_predictions():
    s3 = boto3.client("s3")
    
    response = s3.list_objects_v2(
        Bucket=S3_BUCKET,
        Prefix="predictions/fpl_best_by_position_"
    )
    files = [obj["Key"] for obj in response.get("Contents", [])]
    latest = sorted(files)[-1]
    
    obj = s3.get_object(Bucket=S3_BUCKET, Key=latest)
    content = obj["Body"].read().decode("utf-8")
    
    reader = csv.DictReader(io.StringIO(content))
    rows = list(reader)
    
    lines = []
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


def ask_claude(user_message, predictions_context):
    payload = json.dumps({
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 1024,
        "system": (
            "You are an FPL (Fantasy Premier League) assistant. "
            "Answer questions using the prediction data provided. "
            "Be concise and helpful. Always mention player prices "
            "and predicted points when recommending players. "
            "Keep responses under 200 words as this is a Telegram chat."
        ),
        "messages": [
            {
                "role": "user",
                "content": f"FPL data:\n{predictions_context}\n\nQuestion: {user_message}"
            }
        ]
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01"
        }
    )

    with urllib.request.urlopen(req, timeout=30) as response:
        result = json.loads(response.read())
        return result["content"][0]["text"]


def send_telegram_message(chat_id, text):
    payload = json.dumps({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown"
    }).encode()

    req = urllib.request.Request(
        f"{TELEGRAM_API}/sendMessage",
        data=payload,
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(req, timeout=10) as response:
        return json.loads(response.read())


def send_typing_action(chat_id):
    payload = json.dumps({
        "chat_id": chat_id,
        "action": "typing"
    }).encode()

    req = urllib.request.Request(
        f"{TELEGRAM_API}/sendChatAction",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    urllib.request.urlopen(req, timeout=5)


def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))
        
        # extract message from Telegram update
        message = body.get("message", {})
        chat_id = message.get("chat", {}).get("id")
        user_message = message.get("text", "")

        # ignore non-message updates (joins, leaves etc)
        if not chat_id or not user_message:
            return {"statusCode": 200, "body": "ok"}

        # handle /start command
        if user_message == "/start":
            send_telegram_message(
                chat_id,
                "👋 Welcome to the *FPL Predictor Bot*!\n\n"
                "Ask me anything about this gameweek, for example:\n"
                "• Who are the best value midfielders?\n"
                "• Which defenders have the easiest fixtures?\n"
                "• Who should I captain this week?"
            )
            return {"statusCode": 200, "body": "ok"}

        # show typing indicator while processing
        send_typing_action(chat_id)

        # get predictions and ask Claude
        predictions = get_latest_predictions()
        answer = ask_claude(user_message, predictions)

        # send reply back to user
        send_telegram_message(chat_id, answer)

        return {"statusCode": 200, "body": "ok"}

    except Exception as e:
        print(f"Error: {str(e)}")
        if chat_id:
            send_telegram_message(
                chat_id,
                "Sorry, something went wrong. Please try again."
            )
        return {"statusCode": 200, "body": "ok"}