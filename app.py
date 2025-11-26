import sys
import os
import csv
from datetime import datetime

# --- PATH FIX: MUST BE FIRST! ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# --------------------------------

from flask import Flask, jsonify, render_template, Response, request
from ai_engine.engine import get_latest_data

# ----------------------------------------------------
# IMPORTANT FIX: TELL FLASK WHERE TEMPLATES ARE
# ----------------------------------------------------
app = Flask(
    __name__,
    template_folder="templates",
    static_folder="statics"
)

# ----------------------------------------------------
# ✅ SESSION SUMMARY SAVE FUNCTION
# ----------------------------------------------------
def save_session_summary(total_time, attention_percent, avg_ear, head_center_percent, phone_count):
    summary_file = "session_summary.csv"

    file_empty = not os.path.exists(summary_file) or os.stat(summary_file).st_size == 0

    with open(summary_file, mode="a", newline="") as file:
        writer = csv.writer(file)

        if file_empty:
            writer.writerow([
                "date",
                "total_time_minutes",
                "attention_percentage",
                "avg_EAR",
                "head_center_percentage",
                "phone_detected_times"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_time,
            attention_percent,
            avg_ear,
            head_center_percent,
            phone_count
        ])

# ----------------------------------------------------
# 📌 API → Called when session ends (frontend will call)
# ----------------------------------------------------
@app.route("/end_session", methods=["POST"])
def end_session():
    data = request.json

    total_time        = data.get("total_time", 0)
    attention_percent = data.get("attention", 0)
    avg_ear           = data.get("avgEAR", 0)
    head_center       = data.get("headCenterPercent", 0)
    phone_count       = data.get("phoneCount", 0)

    save_session_summary(
        total_time,
        attention_percent,
        avg_ear,
        head_center,
        phone_count
    )

    return jsonify({"status": "success", "message": "Session summary saved!"})


# ----------------------------------------------------
# FRONT-END ROUTES
# ----------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# --- LIVE DATA API ---
@app.route("/live_data")
def live_data():
    data = get_latest_data()
    return jsonify(data)


# --- LIVE VIDEO STREAM ---
@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            if os.path.exists("latest_frame.jpg"):
                with open("latest_frame.jpg", "rb") as f:
                    frame = f.read()
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" +
                           frame + b"\r\n")

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ----------------------------------------------------
# RUN APP
# ----------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
