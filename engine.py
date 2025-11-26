import pandas as pd
import os
import json
import time

# ------------------------------------------
# FIX: Always use the absolute path to session_log.csv
# ------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SESSION_FILE = os.path.join(BASE_DIR, "session_log.csv")
EXPORT_FILE = os.path.join(BASE_DIR, "session_export.json")


def get_latest_data():
    """Read latest attention data live."""
    if not os.path.exists(SESSION_FILE):
        return {"error": "No session log found"}

    try:
        df = pd.read_csv(SESSION_FILE)
        if df.empty:
            return {"error": "Session file empty"}

        last_row = df.iloc[-1]

        return {
            "timestamp": str(last_row["timestamp"]),
            "ear": float(last_row["ear"]),
            "head": str(last_row["head"]),
            "phone": str(last_row["phone"]),
            "score": int(last_row["score"]),
            "status": str(last_row["status"]),
            "student": os.getenv("STUDENT_NAME", "Unknown")
        }

    except Exception as e:
        return {"error": str(e)}


def get_summary():
    """Return total session summary."""
    if not os.path.exists(SESSION_FILE):
        return {"error": "No session file"}

    df = pd.read_csv(SESSION_FILE)
    total_rows = len(df)
    att_rows = len(df[df["status"] == "ATTENTIVE"])

    if total_rows == 0:
        return {"error": "No data recorded"}

    return {
        "total_seconds": total_rows,
        "attentive_seconds": att_rows,
        "attention_percent": round(att_rows / total_rows * 100, 2)
    }


def export_json():
    """Export latest session as JSON for dashboard."""
    data = get_latest_data()
    summary = get_summary()

    export = {
        "live": data,
        "summary": summary
    }

    with open(EXPORT_FILE, "w") as f:
        json.dump(export, f, indent=4)

    return export


if __name__ == "_main_":
    while True:
        print(export_json())
        time.sleep(1)