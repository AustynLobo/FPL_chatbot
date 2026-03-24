import requests
import json
import boto3, os

CACHE_DIR       = os.path.join("data", "cache")
CACHE_META      = os.path.join(CACHE_DIR, "meta.json")
local_path = CACHE_META

def get_and_sync_fpl_metadata(bucket):
    # 1. Fetch data from FPL
    r = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
    
    # 2. Find the current and next gameweek
    current_gw = next((event for event in r['events'] if event['is_current']), None)
    next_gw_id = (current_gw['id'] + 1) if current_gw else 1

    # 3. Find the deadline for the next gameweek
    next_gw_data = next((event for event in r['events'] if event['id'] == next_gw_id), None)
    
    if not next_gw_data:
        return None

    # Create the metadata dictionary
    meta_content = {
        "last_finished_gw": current_gw['id'] if current_gw else 0,
        "next_gw": next_gw_id,
        "next_deadline_unix": next_gw_data['deadline_time_epoch'],
        "deadline_human": next_gw_data['deadline_time']
    }

    # Save locally
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, 'w') as f:
        json.dump(meta_content, f, indent=4)

    # RETURN the data so we can use it for the S3 export
    return meta_content