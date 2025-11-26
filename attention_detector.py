import cv2
import mediapipe as mp
import numpy as np
import time
from ultralytics import YOLO
from playsound import playsound
import threading
import csv
from datetime import datetime
import os
import pandas as pd
import shutil


# --------------------------------------------
# COLORS & FONTS
# --------------------------------------------
COLOR_BG_DARK = (40, 44, 52)
COLOR_ACCENT_GREEN = (127, 255, 0)
COLOR_ACCENT_RED = (80, 80, 255)
COLOR_ACCENT_CYAN = (255, 255, 0)
COLOR_TEXT_WHITE = (245, 245, 245)
COLOR_TEXT_GRAY = (180, 180, 180)

FONT_MAIN = cv2.FONT_HERSHEY_DUPLEX
FONT_BOLD = cv2.FONT_HERSHEY_TRIPLEX


# --------------------------------------------
# MOVE OLD LOGS TO ARCHIVE
# --------------------------------------------
archive_folder = "session_logs_archive"
if not os.path.exists(archive_folder):
    os.mkdir(archive_folder)

for file in os.listdir():
    if (
        file.startswith("session_log_") and file.endswith(".csv")
        or file.endswith("_session_log.xlsx")
        or file.endswith("_summary.txt")
    ):
        try:
            shutil.move(file, archive_folder)
        except:
            pass


# --------------------------------------------
# ASK STUDENT NAME
# --------------------------------------------
student_name = input("Enter Student Name: ").strip()
if student_name == "":
    student_name = "Unknown_Student"

os.environ["STUDENT_NAME"] = student_name
safe_name = student_name.replace(" ", "_")

print(f"\nTracking for: {student_name}\n")


# --------------------------------------------
# CREATE NEW CSV
# --------------------------------------------
with open("session_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "ear", "head", "phone", "score", "status"])


# --------------------------------------------
# ALERT SOUND
# --------------------------------------------
alert_active = False

def play_alert():
    global alert_active
    if not alert_active:
        alert_active = True
        try:
            playsound("alert.wav")
        except:
            pass
        alert_active = False


# --------------------------------------------
# MODELS
# --------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

phone_model = YOLO("yolov8n.onnx", task="detect")


# --------------------------------------------
# CONSTANTS
# --------------------------------------------
EAR_THRESHOLD = 0.18
PHONE_DROP = 15
attention_score = 100

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

blink_start = time.time()
head_turn_start = time.time()
distraction_start = None
last_log = time.time()
session_start = time.time()


# --------------------------------------------
# EAR FUNCTION
# --------------------------------------------
def calc_EAR(lm, pts, w, h):
    pts_list = [(int(lm[i].x * w), int(lm[i].y * h)) for i in pts]
    A = np.linalg.norm(np.array(pts_list[1]) - np.array(pts_list[5]))
    B = np.linalg.norm(np.array(pts_list[2]) - np.array(pts_list[4]))
    C = np.linalg.norm(np.array(pts_list[0]) - np.array(pts_list[3]))
    return (A + B) / (2 * C)


# --------------------------------------------
# UI COMPONENTS
# --------------------------------------------
def draw_glass_panel(img, x, y, w, h, color=COLOR_BG_DARK, alpha=0.6):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)


def draw_modern_meter(img, score, x, y, radius=60):
    overlay = img.copy()
    cv2.circle(overlay, (x, y), radius, (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

    cv2.circle(img, (x, y), radius, (60, 60, 60), 8)

    color = COLOR_ACCENT_GREEN if score > 50 else COLOR_ACCENT_RED
    end_angle = int((score / 100) * 360)
    cv2.ellipse(img, (x, y), (radius, radius), -90, 0, end_angle, color, 8)

    size = cv2.getTextSize(str(score), FONT_BOLD, 1.2, 2)[0]
    cv2.putText(img, str(score), (x - size[0] // 2, y + size[1] // 2),
                FONT_BOLD, 1.2, COLOR_TEXT_WHITE, 2)


def draw_modern_ui(frame, ear, head, status, score, phone, student_name):
    h, w, _ = frame.shape

    # STATUS BAR
    status_color = COLOR_ACCENT_GREEN if status == "ATTENTIVE" else COLOR_ACCENT_RED
    cv2.rectangle(frame, (w // 2 - 120, 20), (w // 2 + 120, 70), status_color, -1)
    cv2.putText(frame, status, (w // 2 - 90, 58), FONT_BOLD, 0.9, (255, 255, 255), 2)

    # SIDEBAR
    panel_x = w - 300
    draw_glass_panel(frame, panel_x, 0, 300, h)

    cv2.putText(frame, "STUDENT", (panel_x + 20, 50), FONT_MAIN, 0.6, COLOR_TEXT_GRAY)
    cv2.putText(frame, student_name, (panel_x + 20, 85), FONT_BOLD, 0.8, COLOR_TEXT_WHITE)

    draw_modern_meter(frame, score, panel_x + 150, 220, 70)
    cv2.putText(frame, "ATTENTION SCORE", (panel_x + 40, 320), FONT_MAIN, 0.6, COLOR_TEXT_GRAY)

    cv2.putText(frame, f"HEAD: {head}", (panel_x + 20, 380), FONT_BOLD, 0.8, COLOR_ACCENT_CYAN)
    cv2.putText(frame, f"EAR : {ear:.2f}", (panel_x + 20, 450), FONT_BOLD, 0.8, COLOR_ACCENT_CYAN)

    phone_txt = "DETECTED" if phone else "SAFE"
    phone_color = COLOR_ACCENT_RED if phone else COLOR_ACCENT_GREEN
    cv2.putText(frame, f"PHONE: {phone_txt}", (panel_x + 20, 520), FONT_BOLD, 0.8, phone_color)

    return frame


# --------------------------------------------
# MAIN LOOP
# --------------------------------------------
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

cv2.namedWindow("AI Attention Detector", cv2.WINDOW_NORMAL)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    attentive = True
    phone = False
    head = "Center"
    now = time.time()
    ear_avg = 0

    # ---------------- FACE MESH ----------------
    if result.multi_face_landmarks:
        lm = result.multi_face_landmarks[0].landmark

        left = calc_EAR(lm, LEFT_EYE, w, h)
        right = calc_EAR(lm, RIGHT_EYE, w, h)
        ear_avg = (left + right) / 2

        if ear_avg < EAR_THRESHOLD:
            if now - blink_start > 1.2:
                attentive = False
        else:
            blink_start = now

        nx = lm[1].x
        lx = lm[234].x
        rx = lm[454].x

        if nx < lx:
            head = "Right"; attentive = False
        elif nx > rx:
            head = "Left"; attentive = False

        if head != "Center" and now - head_turn_start > 1.7:
            attentive = False
        else:
            head_turn_start = now

    else:
        attentive = False

    # ---------------- PHONE DETECTION ----------------
    results = phone_model(frame, verbose=False, conf=0.5, classes=[67])
    for obj in results[0].boxes:
        x1, y1, x2, y2 = obj.xyxy[0]
        if (x2 - x1) * (y2 - y1) > 4000:
            phone = True
            attentive = False
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          COLOR_ACCENT_RED, 3)

    # ---------------- SCORE ----------------
    attention_score += 1 if attentive else -5
    if phone:
        attention_score -= PHONE_DROP
    attention_score = max(0, min(attention_score, 100))

    status = "ATTENTIVE" if attentive else "NOT ATTENTIVE"

    # ---------------- ALERT ----------------
    if status == "NOT ATTENTIVE":
        if distraction_start is None:
            distraction_start = now
        elif now - distraction_start > 3:
            threading.Thread(target=play_alert).start()
    else:
        distraction_start = None

    # ---------------- LOGGING ----------------
    if now - last_log >= 1:
        with open("session_log.csv", "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                round(ear_avg, 3), head,
                "YES" if phone else "NO",
                attention_score, status
            ])
        last_log = now

    # ---------------- UI DRAW ----------------
    frame = draw_modern_ui(frame, ear_avg, head, status, attention_score, phone, student_name)

    # Save frame for dashboard feed
    cv2.imwrite("latest_frame.jpg", frame)

    cv2.imshow("AI Attention Detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()


# --------------------------------------------
# FINAL REPORT
# --------------------------------------------
df = pd.read_csv("session_log.csv")
total_rows = len(df)
att_rows = len(df[df["status"] == "ATTENTIVE"])

total_time = int(time.time() - session_start)
att_time = att_rows
final_percent = (att_time / total_time * 100) if total_time > 0 else 0

print("\n------- FINAL ATTENTION REPORT -------")
print(f"Student: {student_name}")
print(f"Total Time     : {total_time}s")
print(f"Attentive Time : {att_time}s")
print(f"Attention %    : {final_percent:.2f}%")
print("--------------------------------------")

df.to_excel(f"{safe_name}_session_log.xlsx", index=False)

with open(f"{safe_name}_summary.txt", "w") as f:
    f.write("FINAL ATTENTION REPORT\n")
    f.write("=========================\n")
    f.write(f"Student: {student_name}\n")
    f.write(f"Total Time: {total_time}s\n")
    f.write(f"Attentive: {att_time}s\n")
    f.write(f"Attention %: {final_percent:.2f}%\n")

print("Saved Excel & Summary.")
