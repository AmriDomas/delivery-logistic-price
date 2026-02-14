import requests
import os

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO = "AmriDomas/delivery-logistic-price"

def trigger_retrain():
    if not GITHUB_TOKEN:
        print("[WARN] GITHUB_TOKEN not set, retrain skipped")
        return

    requests.post(
        f"https://api.github.com/repos/{REPO}/dispatches",
        headers={
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json"
        },
        json={"event_type": "drift_detected"}
    )
