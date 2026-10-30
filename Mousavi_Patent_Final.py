# ============================================================
# INTERVIEW MONITOR — PART 1
# CORE SENSORS + FEATURE EXTRACTION + FUSION
# ============================================================

import cv2
import numpy as np
import sounddevice as sd
import threading
import time
import json
from collections import deque, Counter
import os
STOP_FLAG_FILE = "STOP_FLAG.txt"

import sys
import os
import os
import sys

BASE_DIR = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.abspath(".")
STOP_FLAG_FILE = os.path.join(BASE_DIR, "STOP_FLAG.txt")

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS   # PyInstaller temp folder
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


import signal
import sys

GLOBAL_RUNNING = True

def shutdown_handler(signum, frame):
    global GLOBAL_RUNNING
    print("Shutdown signal received")
    GLOBAL_RUNNING = False



# ======================= CONFIG =======================

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
DISPLAY_WIDTH  = 2560
DISPLAY_HEIGHT = 1440

AUDIO_FS = 16000
AUDIO_BLOCKSIZE = 1024
AUDIO_SHORT_WINDOW = int(AUDIO_FS * 0.05)

MOUTH_SMOOTH_ALPHA = 0.8
AUDIO_GATE_THRESHOLD = 0.02

STRESS_THRESHOLD = 0.58
FOCUS_THRESHOLD  = 0.65
CONF_THRESHOLD   = 0.65
STATUS_SHOW_SECONDS = 3.0

SAVE_EVERY_N_FRAMES = 3
SAVE_FULL_FEATURES_EVERY_N_FRAMES = 5

# ======================= AUDIO =======================

audio_buffer = deque(maxlen=AUDIO_FS)
audio_lock = threading.Lock()

def audio_callback(indata, frames, time_info, status):
    mono = indata[:, 0].copy()
    with audio_lock:
        audio_buffer.extend(mono)

def start_audio_stream():
    stream = sd.InputStream(
        samplerate=AUDIO_FS,
        channels=1,
        blocksize=AUDIO_BLOCKSIZE,
        callback=audio_callback
    )
    stream.start()
    return stream

def compute_audio_features():
    with audio_lock:
        if len(audio_buffer) < AUDIO_SHORT_WINDOW:
            return 0.0, 0.0
        samples = np.array(list(audio_buffer)[-AUDIO_SHORT_WINDOW:], dtype=np.float32)

    max_abs = np.max(np.abs(samples))
    if max_abs > 0:
        samples /= max_abs

    energy = float(np.sqrt(np.mean(samples ** 2)))

    corr = np.correlate(samples, samples, mode="full")
    corr = corr[len(corr)//2:]

    min_lag = int(AUDIO_FS / 400)
    max_lag = int(AUDIO_FS / 80)
    corr[:min_lag] = 0.0

    if max_lag >= len(corr):
        return energy, 0.0

    lag = np.argmax(corr[min_lag:max_lag]) + min_lag
    pitch = AUDIO_FS / lag if lag > 0 else 0.0

    return energy, float(pitch)

# ======================= FACE / EYE CASCADES =======================

import sys
import os

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS   # PyInstaller temp folder
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

face_cascade_path = resource_path("haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(face_cascade_path)

if face_cascade.empty():
    raise RuntimeError(f"FATAL: Face cascade failed to load:\n{face_cascade_path}")


eye_cascade_path = resource_path("haarcascade_eye.xml")
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

if eye_cascade.empty():
    raise RuntimeError(f"FATAL: Eye cascade failed to load:\n{eye_cascade_path}")


def get_main_face(gray):
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda r: r[2] * r[3])

# ======================= EYE + GAZE =======================

def compute_eye_activity_and_gaze(gray, face_rect):
    x, y, w, h = face_rect
    eye_region = gray[y:y + int(h * 0.5), x:x + w]
    if eye_region.size == 0:
        return 0.0, "Center", None

    eyes = eye_cascade.detectMultiScale(
        eye_region, 1.1, 3, minSize=(15, 15)
    )

    if len(eyes) == 0:
        return 0.7, "Center", None

    intensities = []
    for (ex, ey, ew, eh) in eyes:
        roi_eye = eye_region[ey:ey + eh, ex:ex + ew]
        intensities.append(np.mean(roi_eye) / 255.0)

    eye_activity = 1.0 - float(np.mean(intensities))
    eye_activity = float(np.clip(eye_activity, 0.0, 1.0))

    col_mean = np.mean(eye_region, axis=0)
    row_mean = np.mean(eye_region, axis=1)
    cx = np.argmin(col_mean) / len(col_mean)
    cy = np.argmin(row_mean) / len(row_mean)

    if cx < 0.35:
        gaze = "Left"
    elif cx > 0.65:
        gaze = "Right"
    elif cy < 0.35:
        gaze = "Up"
    elif cy > 0.65:
        gaze = "Down"
    else:
        gaze = "Center"

    return eye_activity, gaze, eyes

# ======================= MOUTH =======================

def compute_mouth_geometry(gray, face_rect, prev_distance):
    x, y, w, h = face_rect

    y1 = int(y + h * 0.65)
    y2 = int(y + h * 0.80)
    x1 = int(x + w * 0.30)
    x2 = int(x + w * 0.70)

    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0, prev_distance, None

    roi = cv2.GaussianBlur(roi, (5, 5), 0)
    _, thresh = cv2.threshold(
        roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    vertical_sum = np.sum(thresh, axis=1)
    distance = float(np.mean(vertical_sum) / 255.0)

    if prev_distance is None:
        return 0.0, distance, (x1, y1, x2 - x1, y2 - y1)

    diff = abs(distance - prev_distance)
    mouth_act = float(np.clip(diff * 10.0, 0.0, 1.0))

    return mouth_act, distance, (x1, y1, x2 - x1, y2 - y1)

# ======================= CHEST =======================

def compute_chest_flow(prev_gray, gray, face_rect):
    if prev_gray is None:
        return 0.0, None

    x, y, w, h = face_rect

    chest_x1 = x
    chest_x2 = x + w
    chest_y1 = int(y + h * 1.05)
    chest_y2 = int(y + h * 1.65)
    chest_y2 = min(chest_y2, gray.shape[0])

    roi1 = prev_gray[chest_y1:chest_y2, chest_x1:chest_x2]
    roi2 = gray[chest_y1:chest_y2, chest_x1:chest_x2]

    if roi1.size == 0 or roi2.size == 0:
        return 0.0, None

    try:
        roi1s = cv2.resize(roi1, (0, 0), fx=0.5, fy=0.5)
        roi2s = cv2.resize(roi2, (0, 0), fx=0.5, fy=0.5)

        flow = cv2.calcOpticalFlowFarneback(
            roi1s, roi2s, None,
            0.5, 2, 9, 2, 5, 1.1, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        chest_val = float(np.clip(np.mean(mag) * 10.0, 0.0, 1.0))
    except Exception:
        chest_val = 0.0

    return chest_val, (chest_x1, chest_y1,
                       chest_x2 - chest_x1, chest_y2 - chest_y1)

# ======================= FUSION =======================

def compute_states(mouth, chest, eye, audio_energy, gaze):
    energy_n = float(np.clip(audio_energy * 6.0, 0.0, 1.0))
    gaze_penalty = 0.0 if gaze == "Center" else 0.08

    stress = float(np.clip(
        0.30 * mouth +
        0.30 * chest +
        0.25 * energy_n +
        0.10 * eye +
        0.05 * gaze_penalty,
        0.0, 1.0
    ))

    gaze_center_score = 1.0 if gaze == "Center" else 0.75

    focus = float(np.clip(
        0.35 * gaze_center_score +
        0.35 * (1.0 - eye) +
        0.30 * (1.0 - chest),
        0.0, 1.0
    ))

    confidence = float(np.clip(
        0.6 * (1.0 - stress) + 0.4 * focus,
        0.0, 1.0
    ))

    if stress > 0.75 and focus < 0.45:
        emotion = "Anxious"
    elif stress > 0.75:
        emotion = "Stressed"
    elif confidence > 0.75 and focus > 0.55:
        emotion = "Confident"
    elif focus > 0.65:
        emotion = "Focused"
    elif focus < 0.35 and stress < 0.35:
        emotion = "Bored"
    elif stress > 0.45 and focus > 0.45:
        emotion = "Engaged"
    elif focus < 0.45 and stress > 0.35:
        emotion = "Distracted"
    else:
        emotion = "Neutral"

    return stress, focus, confidence, emotion
# ============================================================
# INTERVIEW MONITOR — PART 2
# CRASH-SAFE MULTI-FORMAT WRITERS (TXT + JSON + CSV + TSV)
# ============================================================

import csv
import os

# ======================= OUTPUT FILES =======================

# ---- Reduced fused logs (1/3) ----
REDUCED_TXT  = "session_log_reduced.txt"
REDUCED_JSON = "session_log_reduced.json"
REDUCED_CSV  = "session_log_reduced.csv"
REDUCED_TSV  = "session_log_reduced.tsv"

# ---- Final summary ----
SUMMARY_TXT  = "final_summary.txt"
SUMMARY_JSON = "final_summary.json"
SUMMARY_CSV  = "final_summary.csv"
SUMMARY_TSV  = "final_summary.tsv"

# ---- Per-modality 1/5 light logs ----
MOUTH_TXT  = "mouth_features_1of5.txt"
MOUTH_JSON = "mouth_features_1of5.json"
MOUTH_CSV  = "mouth_features_1of5.csv"
MOUTH_TSV  = "mouth_features_1of5.tsv"

CHEST_TXT  = "chest_features_1of5.txt"
CHEST_JSON = "chest_features_1of5.json"
CHEST_CSV  = "chest_features_1of5.csv"
CHEST_TSV  = "chest_features_1of5.tsv"

EYE_TXT  = "eye_features_1of5.txt"
EYE_JSON = "eye_features_1of5.json"
EYE_CSV  = "eye_features_1of5.csv"
EYE_TSV  = "eye_features_1of5.tsv"

AUDIO_TXT  = "audio_features_1of5.txt"
AUDIO_JSON = "audio_features_1of5.json"
AUDIO_CSV  = "audio_features_1of5.csv"
AUDIO_TSV  = "audio_features_1of5.tsv"


# ======================= INIT FUNCTIONS =======================

def init_json_file(path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump([], f)

def init_text_file(path, header=None):
    with open(path, "w", encoding="utf-8") as f:
        if header:
            f.write(header + "\n")

def init_csv_tsv(path, header_list, sep):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=sep)
        writer.writerow(header_list)


def init_all_output_files():
    # -------- Reduced headers --------
    reduced_header = [
        "timestamp","mouth","chest","eye","gaze",
        "audio_energy","pitch",
        "stress","focus","confidence","emotion"
    ]

    init_text_file(REDUCED_TXT, "\t".join(reduced_header))
    init_json_file(REDUCED_JSON)
    init_csv_tsv(REDUCED_CSV, reduced_header, ",")
    init_csv_tsv(REDUCED_TSV, reduced_header, "\t")

    # -------- Per-modality headers --------
    mouth_header = ["timestamp","smoothed_mouth","raw_mouth"]
    chest_header = ["timestamp","chest_motion"]
    eye_header   = ["timestamp","eye_activity","gaze"]
    audio_header = ["timestamp","audio_energy","pitch"]

    for txt, js, csvf, tsvf, hdr in [
        (MOUTH_TXT,  MOUTH_JSON,  MOUTH_CSV,  MOUTH_TSV,  mouth_header),
        (CHEST_TXT,  CHEST_JSON,  CHEST_CSV,  CHEST_TSV,  chest_header),
        (EYE_TXT,    EYE_JSON,    EYE_CSV,    EYE_TSV,    eye_header),
        (AUDIO_TXT,  AUDIO_JSON,  AUDIO_CSV,  AUDIO_TSV,  audio_header)
    ]:
        init_text_file(txt, "\t".join(hdr))
        init_json_file(js)
        init_csv_tsv(csvf, hdr, ",")
        init_csv_tsv(tsvf, hdr, "\t")


# ======================= APPEND HELPERS =======================

def append_json(path, row_dict):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.append(row_dict)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def append_txt(path, row_list):
    with open(path, "a", encoding="utf-8") as f:
        f.write("\t".join(str(v) for v in row_list) + "\n")

def append_csv_tsv(path, row_list, sep):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=sep)
        writer.writerow(row_list)


# ======================= PER-MODALITY SAVE (1/5) =======================

def save_modality_1of5(timestamp, smoothed_mouth, mouth, chest, eye, gaze, audio_energy, pitch):

    mouth_row = [timestamp, smoothed_mouth, mouth]
    chest_row = [timestamp, chest]
    eye_row   = [timestamp, eye, gaze]
    audio_row = [timestamp, audio_energy, pitch]

    append_txt(MOUTH_TXT, mouth_row)
    append_txt(CHEST_TXT, chest_row)
    append_txt(EYE_TXT, eye_row)
    append_txt(AUDIO_TXT, audio_row)

    append_csv_tsv(MOUTH_CSV, mouth_row, ",")
    append_csv_tsv(CHEST_CSV, chest_row, ",")
    append_csv_tsv(EYE_CSV, eye_row, ",")
    append_csv_tsv(AUDIO_CSV, audio_row, ",")

    append_csv_tsv(MOUTH_TSV, mouth_row, "\t")
    append_csv_tsv(CHEST_TSV, chest_row, "\t")
    append_csv_tsv(EYE_TSV, eye_row, "\t")
    append_csv_tsv(AUDIO_TSV, audio_row, "\t")

    append_json(MOUTH_JSON,  dict(zip(["timestamp","smoothed_mouth","raw_mouth"], mouth_row)))
    append_json(CHEST_JSON,  dict(zip(["timestamp","chest_motion"], chest_row)))
    append_json(EYE_JSON,    dict(zip(["timestamp","eye_activity","gaze"], eye_row)))
    append_json(AUDIO_JSON,  dict(zip(["timestamp","audio_energy","pitch"], audio_row)))


# ======================= REDUCED FUSED SAVE (1/3) =======================

def save_reduced_1of3(timestamp, smoothed_mouth, chest, eye, gaze, audio_energy, pitch,
                      stress, focus, confidence, emotion):

    row = [
        timestamp,
        smoothed_mouth, chest, eye, gaze,
        audio_energy, pitch,
        stress, focus, confidence, emotion
    ]

    append_txt(REDUCED_TXT, row)
    append_csv_tsv(REDUCED_CSV, row, ",")
    append_csv_tsv(REDUCED_TSV, row, "\t")

    append_json(REDUCED_JSON, {
        "timestamp":timestamp,
        "mouth":smoothed_mouth,
        "chest":chest,
        "eye":eye,
        "gaze":gaze,
        "audio_energy":audio_energy,
        "pitch":pitch,
        "stress":stress,
        "focus":focus,
        "confidence":confidence,
        "emotion":emotion
    })


# ======================= FINAL SUMMARY SAVE =======================

def save_final_summary(summary_rows):

    # TXT
    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        for m, v in summary_rows:
            f.write(f"{m}: {v}\n")

    # CSV + TSV
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric","Value"])
        writer.writerows(summary_rows)

    with open(SUMMARY_TSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["Metric","Value"])
        writer.writerows(summary_rows)

    # JSON
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
# ============================================================
# INTERVIEW MONITOR — PART 3
# MAIN LOOP + UI + FINAL SUMMARY + SAFE SHUTDOWN
# ============================================================


signal.signal(signal.SIGTERM, shutdown_handler)
signal.signal(signal.SIGINT, shutdown_handler)



def main():

    # -------- INIT ALL OUTPUT FILES (CRASH SAFE) --------
    init_all_output_files()

    # -------- Start audio safely --------
    audio_stream = start_audio_stream()

    # -------- Start camera --------
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    prev_gray = None
    prev_mouth_distance = None
    smoothed_mouth = 0.0
    frame_index = 0

    # -------- Storage for FINAL summary --------
    all_stress = []
    all_focus = []
    all_confidence = []
    all_emotions = []

    # -------- FPS --------
    fps_last_time = time.time()
    fps_count = 0
    fps_display = 0.0

    print("SYSTEM STARTED — PRESS Q OR ESC TO STOP")

    # ======================= LOOP =======================
    while True:
        if os.path.exists(STOP_FLAG_FILE):
            break

        ret, frame = cap.read()
        if not ret:
            break

        # frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        now = time.time()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))

        face = get_main_face(gray)

        mouth = chest = eye = 0.0
        gaze = "Center"
        mouth_box = chest_box = None
        eye_boxes = None

        if face is not None:
            x, y, w, h = face

            # Face box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)

            # Mouth
            mouth, prev_mouth_distance, mouth_box = compute_mouth_geometry(
                gray, face, prev_mouth_distance
            )

            # Chest
            chest, chest_box = compute_chest_flow(prev_gray, gray, face)

            # Eyes & gaze
            eye, gaze, eye_boxes = compute_eye_activity_and_gaze(gray, face)

        # -------- Audio --------
        audio_energy, pitch = compute_audio_features()

        # -------- Gate mouth by audio --------
        mouth_for_fusion = mouth if audio_energy >= AUDIO_GATE_THRESHOLD else 0.0

        # -------- Smooth mouth --------
        smoothed_mouth = (
            MOUTH_SMOOTH_ALPHA * smoothed_mouth +
            (1.0 - MOUTH_SMOOTH_ALPHA) * mouth_for_fusion
        )

        # -------- Fusion --------
        stress, focus, confidence, emotion = compute_states(
            smoothed_mouth, chest, eye, audio_energy, gaze
        )

        # -------- FINAL SUMMARY BUFFERS --------
        all_stress.append(stress)
        all_focus.append(focus)
        all_confidence.append(confidence)
        all_emotions.append(emotion)

        # -------- 1/5 PER-MODALITY SAVE --------
        if frame_index % SAVE_FULL_FEATURES_EVERY_N_FRAMES == 0:
            save_modality_1of5(
                timestamp,
                smoothed_mouth,
                mouth,
                chest,
                eye,
                gaze,
                audio_energy,
                pitch
            )

        # -------- 1/3 REDUCED FUSED SAVE --------
        if frame_index % SAVE_EVERY_N_FRAMES == 0:
            save_reduced_1of3(
                timestamp,
                smoothed_mouth,
                chest,
                eye,
                gaze,
                audio_energy,
                pitch,
                stress,
                focus,
                confidence,
                emotion
            )

        # -------- FPS --------
        fps_count += 1
        if now - fps_last_time >= 1.0:
            fps_display = fps_count / (now - fps_last_time)
            fps_last_time = now
            fps_count = 0

        # -------- OVERLAY TEXT --------
        overlay_lines = [
            f"Time: {timestamp}",
            f"FPS: {fps_display:.1f}",
            f"Mouth: {smoothed_mouth:.2f}",
            f"Chest: {chest:.2f}",
            f"Eye: {eye:.2f} | Gaze: {gaze}",
            f"Audio Energy: {audio_energy:.3f}",
            f"Pitch: {pitch:.1f} Hz",
            f"Stress: {stress:.2f}",
            f"Focus: {focus:.2f}",
            f"Confidence: {confidence:.2f}",
            f"Emotion: {emotion}"
        ]

        y = 25
        for line in overlay_lines:

            # Defaults: small yellow text
            color = (0, 255, 255)
            scale = 0.5
            thick = 1

            # Big colored four
            if line.startswith("Stress"):
                color = (0, 0, 255)
                scale = 0.8
                thick = 3
                y += 20
            elif line.startswith("Confidence"):
                color = (0, 255, 0)
                scale = 0.8
                thick = 3
            elif line.startswith("Focus"):
                color = (255, 0, 0)
                scale = 0.8
                thick = 3
            elif line.startswith("Emotion"):
                color = (0, 140, 255)
                scale = 0.8
                thick = 3

            cv2.putText(
                frame, line,
                (15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color, thick,
                cv2.LINE_AA
            )

            y += 30 if scale > 0.5 else 17


        # -------- BOXES --------
        if mouth_box is not None:
            mx, my, mw, mh = mouth_box
            cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (255, 0, 0), 2)

        if eye_boxes is not None and face is not None:
            fx, fy, fw, fh = face
            for (ex, ey, ew, eh) in eye_boxes:
                cv2.rectangle(
                    frame,
                    (fx + ex, fy + ey),
                    (fx + ex + ew, fy + ey + eh),
                    (255, 255, 0), 2
                )

        if chest_box is not None:
            cx, cy, cw, ch = chest_box
            cv2.rectangle(
                frame,
                (cx, cy),
                (cx + cw, cy + ch),
                (255, 0, 255), 2
            )

        # -------- SHOW --------
        # frame_big = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        # cv2.imshow("Interview Monitor - FINAL", frame_big)
        frame_up = cv2.resize(
            frame, None,
            fx=2.0, fy=1.5,
            interpolation=cv2.INTER_LINEAR
        )
        
        cv2.namedWindow("Interview Monitor - FINAL", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Interview Monitor - FINAL", frame_up)




        prev_gray = gray.copy()
        frame_index += 1

        key = cv2.waitKey(1) & 0xFF
        if key in [27, ord("q")]:
            break

    # ======================= CLEANUP =======================
    if os.path.exists(STOP_FLAG_FILE):
        os.remove(STOP_FLAG_FILE)

    cap.release()
    cv2.destroyAllWindows()
    audio_stream.stop()

    print("SYSTEM STOPPED — COMPUTING FINAL SUMMARY")

    # ======================= FINAL SUMMARY =======================
    if len(all_stress) > 0:
        mean_stress = float(np.mean(all_stress))
        mean_focus  = float(np.mean(all_focus))
        mean_conf   = float(np.mean(all_confidence))
    else:
        mean_stress = mean_focus = mean_conf = 0.0

    counts = Counter(all_emotions)
    total = sum(counts.values()) if counts else 0

    emotion_percent = {}
    if total > 0:
        for k, v in counts.items():
            emotion_percent[k] = round(100.0 * v / total, 2)
        dominant_emotion = max(emotion_percent, key=emotion_percent.get)
    else:
        dominant_emotion = "None"

    summary_rows = [
        ["Mean Stress", mean_stress],
        ["Mean Focus", mean_focus],
        ["Mean Confidence", mean_conf],
        ["Dominant Emotion", dominant_emotion]
    ]

    for k, v in emotion_percent.items():
        summary_rows.append([f"Emotion % {k}", v])

    save_final_summary(summary_rows)

    print("FINAL SUMMARY SAVED (TXT + JSON + CSV + TSV)")
    print("DOMINANT EMOTION:", dominant_emotion)


# ======================= ENTRY POINT =======================

if __name__ == "__main__":
    main()
