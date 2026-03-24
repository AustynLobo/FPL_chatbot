import boto3
import os
import json
import urllib.request
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

def lambda_handler(event, context):
    # 1. Connect to your existing Chat History table
    dynamodb = boto3.resource('dynamodb')
    history_table = dynamodb.Table(os.environ['HISTORY_TABLE_NAME'])
    
    # 2. Get the FPL Deadline from your S3 cache
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=os.environ['S3_BUCKET'], Key='cache/meta.json')
    meta = json.loads(obj['Body'].read().decode('utf-8'))
    
    current_gw = meta.get('next_gw')
    last_notified = meta.get('last_notified_gw', 0)



    # Use 'deadline_time_epoch' from FPL API if available in your meta.json
    deadline_ts = meta.get('next_deadline_unix') 
    deadline_dt_utc = datetime.fromtimestamp(deadline_ts, tz=timezone.utc)
    german_tz = ZoneInfo("Europe/Berlin")
    deadline_dt_german = deadline_dt_utc.astimezone(german_tz)    
    now = datetime.now(timezone.utc)
    
    # 3. Time Check: Are we in the "12-hour-warning" window?
    hours_to_deadline = (deadline_dt_utc - now).total_seconds() / 3600
    
    # Check if we are between 8 and 14 hours away (prevents missing it if check is every 6h)
    if 2 <= hours_to_deadline <= 6:   
        # 4. Get all unique chat_ids from your history table
        # Pro-tip: Use ProjectionExpression to only pull the ID and save money
        response = history_table.scan(ProjectionExpression="chat_id")
        unique_chats = {item['chat_id'] for item in response.get('Items', [])}
        
        # 5. Send the notifications
        msg = f"*FPL Deadline Alert*\nGW{meta['next_gw']} starts in ~12 hours!\n {deadline_dt_german.strftime('%d %b, %H:%M')}"
        
        for chat_id in unique_chats:
            send_telegram(chat_id, msg)

            # --- Update "Memory" ---
            # Save back to S3 so the next hourly run knows we are DONE for this GW
        meta['last_finished_gw'] = current_gw
        s3.put_object(
            Bucket=os.environ['S3_BUCKET'],
            Key='cache/meta.json',
            Body=json.dumps(meta),
            ContentType='application/json'
        )
        print(f"Notifications sent for GW{current_gw}. S3 updated.")
            
    return {"status": "Complete"}

def send_telegram(chat_id, text):
    token = os.environ['TELEGRAM_TOKEN']
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = json.dumps({"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}).encode()
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    urllib.request.urlopen(req)