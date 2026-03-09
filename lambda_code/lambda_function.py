import json
import boto3
import csv
import io
import urllib.request
import os

S3_BUCKET = "my-fpl-predictions"


def get_latest_predictions():
    s3 = boto3.client("s3")
    
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix="predictions/fpl_best_by_position_")
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
            "and predicted points when recommending players."
        ),
        "messages": [
            {
                "role": "user",
                "content": f"Here is the latest FPL prediction data:\n\n{predictions_context}\n\nQuestion: {user_message}"
            }
        ]
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01"
        }
    )

    with urllib.request.urlopen(req, timeout=30) as response:
        result = json.loads(response.read())
        return result["content"][0]["text"]

def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))
        user_message = body.get("message", "")
        
        if not user_message:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "No message provided"})
            }
        
        predictions = get_latest_predictions()
        answer = ask_claude(user_message, predictions)
        
        return {
            "statusCode": 200,
            "headers": {"Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"response": answer})
        }
    
    except urllib.error.HTTPError as e:
        error_body = e.read().decode()
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": f"HTTP {e.code}",
                "detail": error_body  # this will tell us exactly what Groq is rejecting
            })
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }