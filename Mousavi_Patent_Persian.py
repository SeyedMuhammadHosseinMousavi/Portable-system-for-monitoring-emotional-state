# ============================================================
# INTERVIEW MONITOR — FULL PIPELINE (NO JSON)
# FEATURES + SUB-FEATURES + 6 HIGH-LEVEL METRICS
# SAVES: combined_full / combined_reduced / summary
# FORMATS: TXT, CSV, TSV
# ============================================================

import cv2
import numpy as np
import sounddevice as sd
import threading
import time
import csv
from collections import deque, Counter
# ======= PERSIAN TEXT SUPPORT (NO UI CHANGE) =======
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

PERSIAN_FONT_PATH = "Vazirmatn-Regular.ttf"  # same font as helper

# ============================================================
# CONFIGURATION
# ============================================================

FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
DISPLAY_WIDTH  = 2560
DISPLAY_HEIGHT = 1440

# Camera FPS hint (used only to interpret SAVE_RATE mentally)
CAMERA_FPS = 30.0

AUDIO_FS = 16000
AUDIO_BLOCKSIZE = 1024
AUDIO_SHORT_WINDOW = int(AUDIO_FS * 0.05)   # ~50 ms window

MOUTH_SMOOTH_ALPHA = 0.8
AUDIO_GATE_THRESHOLD = 0.02  # if below: mouth activity gated to 0

# Save every N frames (approx CAMERA_FPS / SAVE_RATE samples per second)
# With CAMERA_FPS ~30 and SAVE_RATE=10 → ~3 samples per second
SAVE_RATE = 10          # combined_full
SAVE_REDUCED_RATE = 10  # combined_reduced

# ============================================================
# AUDIO BUFFER + FEATURE EXTRACTION
# ============================================================

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
    """
    Returns:
        audio_energy          (RMS, 0..1 after normalization)
        audio_pitch           (Hz)
        audio_zero_cross_rate (0..1)
        audio_rms             (same as energy, kept for completeness)
        audio_peak            (max abs amplitude, 0..1)
    """
    with audio_lock:
        if len(audio_buffer) < AUDIO_SHORT_WINDOW:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        samples = np.array(list(audio_buffer)[-AUDIO_SHORT_WINDOW:], dtype=np.float32)

    max_abs = np.max(np.abs(samples))
    if max_abs > 0:
        samples = samples / max_abs * 0.9   # prevent hard saturation


    # RMS energy
    rms = float(np.sqrt(np.mean(samples ** 2)))

    # Zero-crossing rate
    zcr = float(((samples[:-1] * samples[1:]) < 0).sum() / len(samples))

    # Peak amplitude
    peak = float(np.max(np.abs(samples)))

    # Pitch estimation (simple autocorrelation)
    corr = np.correlate(samples, samples, mode="full")
    corr = corr[len(corr) // 2:]

    min_lag = int(AUDIO_FS / 400)  # ~400 Hz upper frequency
    max_lag = int(AUDIO_FS / 80)   # ~80 Hz lower frequency
    corr[:min_lag] = 0.0

    if max_lag >= len(corr):
        return rms, 0.0, zcr, rms, peak

    lag = np.argmax(corr[min_lag:max_lag]) + min_lag
    pitch = AUDIO_FS / lag if lag > 0 else 0.0

    return rms, float(pitch), zcr, rms, peak


# ============================================================
# FACE / EYE CASCADES
# ============================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)


def get_main_face(gray):
    """Return the largest detected face rectangle (x,y,w,h) or None."""
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda r: r[2] * r[3])


# ============================================================
# EYE + GAZE SUB-FEATURES
# ============================================================

def extract_eye_features(gray, face_rect):
    """
    Returns:
        left_eye_intensity   (0..1)
        right_eye_intensity  (0..1)
        eye_intensity_mean   (0..1)
        eye_activity         (0..1, 1=more activity)
        gaze_center_x        (0..1)
        gaze_center_y        (0..1)
        gaze_label           ("Center","Left","Right","Up","Down")
    """
    if face_rect is None:
        return (
            0.0,    # left_eye_intensity
            0.0,    # right_eye_intensity
            0.0,    # eye_intensity_mean
            0.0,    # eye_activity
            0.5,    # gaze_center_x
            0.5,    # gaze_center_y
            "Center",
            []      # eye_boxes
        )



    x, y, w, h = face_rect
    eye_region = gray[y: y + int(h * 0.5), x: x + w]

    if eye_region.size == 0:
        return (
            0.0,    # left_eye_intensity
            0.0,    # right_eye_intensity
            0.0,    # eye_intensity_mean
            0.0,    # eye_activity
            0.5,    # gaze_center_x
            0.5,    # gaze_center_y
            "Center",
            []      # eye_boxes
        )



    eyes = eye_cascade.detectMultiScale(eye_region, 1.1, 3, minSize=(15, 15))

    if len(eyes) == 0:
        return (
            0.7,
            0.7,
            0.7,
            0.3,
            0.5,
            0.5,
            "Center",
            []
        )


    intensities = []
    left_eye_intensity = 0.0
    right_eye_intensity = 0.0

    for (ex, ey, ew, eh) in eyes:
        roi_eye = eye_region[ey:ey + eh, ex:ex + ew]
        value = float(np.mean(roi_eye) / 255.0)
        intensities.append(value)

        # Rough split
        if ex < w / 2:
            left_eye_intensity = value
        else:
            right_eye_intensity = value

    eye_intensity_mean = float(np.mean(intensities))
    eye_activity = float(np.clip(1.0 - eye_intensity_mean, 0.0, 1.0))

    col_mean = np.mean(eye_region, axis=0)
    row_mean = np.mean(eye_region, axis=1)

    gaze_center_x = float(np.argmin(col_mean) / len(col_mean))
    gaze_center_y = float(np.argmin(row_mean) / len(row_mean))

    if gaze_center_x < 0.35:
        gaze_label = "Left"
    elif gaze_center_x > 0.65:
        gaze_label = "Right"
    elif gaze_center_y < 0.35:
        gaze_label = "Up"
    elif gaze_center_y > 0.65:
        gaze_label = "Down"
    else:
        gaze_label = "Center"

    return (
        left_eye_intensity,
        right_eye_intensity,
        eye_intensity_mean,
        eye_activity,
        gaze_center_x,
        gaze_center_y,
        gaze_label,
        eyes
    )

# ============================================================
# MOUTH SUB-FEATURES
# ============================================================

def extract_mouth_features(gray, face_rect, prev_distance):
    """
    Returns:
        mouth_open_raw   (vertical_sum based opening)
        mouth_change     (abs difference from prev)
        mouth_activity   (0..1)
        mouth_box        ((x,y,w,h) or None)
        new_prev_distance
    """
    if face_rect is None:
        return 0.0, 0.0, 0.0, None, prev_distance

    x, y, w, h = face_rect

    y1 = int(y + h * 0.65)
    y2 = int(y + h * 0.80)
    x1 = int(x + w * 0.30)
    x2 = int(x + w * 0.70)

    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0, 0.0, 0.0, None, prev_distance

    roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
    _, thresh = cv2.threshold(
        roi_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    vertical_sum = np.sum(thresh, axis=1)
    mouth_open_raw = float(np.mean(vertical_sum) / 255.0)

    if prev_distance is None:
        mouth_change = 0.0
        mouth_activity = 0.0
    else:
        mouth_change = abs(mouth_open_raw - prev_distance)
        mouth_activity = float(np.clip(mouth_change * 10.0, 0.0, 1.0))

    mouth_box = (x1, y1, x2 - x1, y2 - y1)
    new_prev_distance = mouth_open_raw

    return mouth_open_raw, mouth_change, mouth_activity, mouth_box, new_prev_distance


# ============================================================
# CHEST SUB-FEATURES
# ============================================================

def extract_chest_features(prev_gray, gray, face_rect):
    """
    Returns:
        chest_flow_raw   (mean optical flow magnitude)
        chest_activity   (0..1, scaled)
        chest_box        ((x,y,w,h) or None)
    """
    if prev_gray is None or face_rect is None:
        return 0.0, 0.0, None

    x, y, w, h = face_rect

    chest_x1 = x
    chest_x2 = x + w
    chest_y1 = int(y + h * 1.05)
    chest_y2 = int(y + h * 1.65)
    
    # Clamp to frame
    chest_y1 = max(0, min(chest_y1, gray.shape[0] - 1))
    chest_y2 = max(chest_y1 + 1, min(chest_y2, gray.shape[0]))


    roi1 = prev_gray[chest_y1:chest_y2, chest_x1:chest_x2]
    roi2 = gray[chest_y1:chest_y2, chest_x1:chest_x2]

    if roi1.size == 0 or roi2.size == 0:
        return 0.0, 0.0, None

    try:
        r1 = cv2.resize(roi1, (0, 0), fx=0.5, fy=0.5)
        r2 = cv2.resize(roi2, (0, 0), fx=0.5, fy=0.5)

        flow = cv2.calcOpticalFlowFarneback(
            r1, r2, None,
            0.5, 2, 9, 2, 5, 1.1, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        chest_flow_raw = float(np.mean(mag))
        chest_activity = float(np.clip(chest_flow_raw / 0.25, 0.0, 1.0))

    except Exception:
        chest_flow_raw = 0.0
        chest_activity = 0.0

    chest_box = (chest_x1, chest_y1, chest_x2 - chest_x1, chest_y2 - chest_y1)
    return chest_flow_raw, chest_activity, chest_box


# ============================================================
# FUSION ENGINE + HIGH-LEVEL METRICS
# ============================================================

def compute_fusion_states(
    mouth_smoothed,
    chest_activity,
    eye_activity,
    audio_energy,
    gaze_label
):
    """
    Returns:
        energy_normalized
        gaze_penalty
        stress
        focus
        confidence
        emotion_label
    """
    energy_normalized = float(np.clip(audio_energy * 3.0, 0.0, 1.0))

    gaze_penalty = 0.0 if gaze_label == "Center" else 0.08

    # STRESS
    stress = float(np.clip(
        0.30 * mouth_smoothed +
        0.30 * chest_activity +
        0.25 * energy_normalized +
        0.10 * eye_activity +
        0.05 * gaze_penalty,
        0.0, 1.0
    ))

    # FOCUS
    gaze_center_score = 1.0 if gaze_label == "Center" else 0.75
    focus = float(np.clip(
        0.35 * gaze_center_score +
        0.35 * (1.0 - eye_activity) +
        0.30 * (1.0 - chest_activity),
        0.0, 1.0
    ))

    # CONFIDENCE
    confidence = float(np.clip(
        0.6 * (1.0 - stress) + 0.4 * focus,
        0.0, 1.0
    ))


    # EMOTION LABEL
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

    return energy_normalized, gaze_penalty, stress, focus, confidence, emotion


def compute_high_level_metrics(
    stress,
    focus,
    confidence,
    eye_activity,
    chest_activity,
    audio_energy_normalized
):
    """
    Returns:
        trustworthiness
        hireability
        mental_resilience
        social_dominance
        psychological_fatigue
        adaptability
    """

    trustworthiness = float(np.clip(
        0.4 * (1.0 - stress) +
        0.3 * focus +
        0.2 * (1.0 - eye_activity) +
        0.1 * (1.0 - audio_energy_normalized),
        0.0, 1.0
    ))

    hireability = float(np.clip(
        0.4 * confidence +
        0.3 * focus +
        0.3 * (1.0 - stress),
        0.0, 1.0
    ))

    mental_resilience = float(np.clip(
        0.5 * (1.0 - stress) +
        0.3 * confidence +
        0.2 * (1.0 - chest_activity),
        0.0, 1.0
    ))

    social_dominance = float(np.clip(
        0.5 * audio_energy_normalized +
        0.3 * confidence +
        0.2 * focus,
        0.0, 1.0
    ))

    psychological_fatigue = float(np.clip(
        0.6 * (1.0 - focus) +
        0.2 * eye_activity +
        0.2 * (1.0 - confidence),
        0.0, 1.0
    ))

    adaptability = float(np.clip(
        0.4 * confidence +
        0.3 * focus +
        0.3 * (1.0 - stress),
        0.0, 1.0
    ))

    return (
        trustworthiness,
        hireability,
        mental_resilience,
        social_dominance,
        psychological_fatigue,
        adaptability
    )


# ============================================================
# FILE OUTPUT SYSTEM (TXT / CSV / TSV)
# ============================================================

COMBINED_TXT = "combined_full.txt"
COMBINED_CSV = "combined_full.csv"
COMBINED_TSV = "combined_full.tsv"

REDUCED_TXT = "combined_reduced.txt"
REDUCED_CSV = "combined_reduced.csv"
REDUCED_TSV = "combined_reduced.tsv"

SUMMARY_TXT = "summary.txt"
SUMMARY_CSV = "summary.csv"
SUMMARY_TSV = "summary.tsv"


def write_txt_header(path, header_list):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header_list) + "\n")


def write_csv_header(path, header_list):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header_list)


def write_tsv_header(path, header_list):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header_list)


def append_txt(path, row_list):
    with open(path, "a", encoding="utf-8") as f:
        f.write("\t".join(str(v) for v in row_list) + "\n")


def append_csv(path, row_list):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row_list)


def append_tsv(path, row_list):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(row_list)


COMBINED_HEADER = [
    "timestamp",

    # Mouth
    "mouth_open_raw",
    "mouth_change",
    "mouth_activity",
    "mouth_smoothed",
    "mouth_final",

    # Chest
    "chest_flow_raw",
    "chest_activity",
    "chest_final",

    # Eyes
    "left_eye_intensity",
    "right_eye_intensity",
    "eye_intensity_mean",
    "eye_activity",
    "eye_final",

    # Gaze
    "gaze_center_x",
    "gaze_center_y",
    "gaze_label",
    "gaze_penalty",
    "gaze_final",

    # Audio
    "audio_energy",
    "audio_pitch",
    "audio_zero_cross_rate",
    "audio_peak",
    "audio_energy_normalized",
    "audio_final",

    # Fusion inputs
    "mouth_for_fusion",
    "energy_normalized",
    "gaze_penalty",

    # Fusion final states
    "stress",
    "focus",
    "confidence",
    "emotion",

    # High-level metrics
    "trustworthiness",
    "hireability",
    "mental_resilience",
    "social_dominance",
    "psychological_fatigue",
    "adaptability"
]



REDUCED_HEADER = [
    "timestamp",
    "mouth_smoothed",
    "chest_activity",
    "eye_activity",
    "gaze_label",
    "audio_energy",
    "audio_pitch",
    "stress",
    "focus",
    "confidence",
    "emotion",
    "trustworthiness",
    "hireability",
    "mental_resilience",
    "social_dominance",
    "psychological_fatigue",
    "adaptability"
]


def init_all_output_files():
    # full
    write_txt_header(COMBINED_TXT, COMBINED_HEADER)
    write_csv_header(COMBINED_CSV, COMBINED_HEADER)
    write_tsv_header(COMBINED_TSV, COMBINED_HEADER)

    # reduced
    write_txt_header(REDUCED_TXT, REDUCED_HEADER)
    write_csv_header(REDUCED_CSV, REDUCED_HEADER)
    write_tsv_header(REDUCED_TSV, REDUCED_HEADER)

    # summary
    write_txt_header(SUMMARY_TXT, ["metric", "value"])
    write_csv_header(SUMMARY_CSV, ["metric", "value"])
    write_tsv_header(SUMMARY_TSV, ["metric", "value"])


def assemble_combined_row(
    timestamp,

    # Mouth
    mouth_open_raw,
    mouth_change,
    mouth_activity,
    mouth_smoothed,
    mouth_final,

    # Chest
    chest_flow_raw,
    chest_activity,
    chest_final,

    # Eyes
    left_eye_intensity,
    right_eye_intensity,
    eye_intensity_mean,
    eye_activity,
    eye_final,

    # Gaze
    gaze_center_x,
    gaze_center_y,
    gaze_label,
    gaze_penalty,
    gaze_final,

    # Audio
    audio_energy,
    audio_pitch,
    audio_zero_cross_rate,
    audio_peak,
    audio_energy_normalized,
    audio_final,

    # Fusion inputs
    mouth_for_fusion,
    energy_normalized,
    # gaze_penalty,

    # Final states
    stress,
    focus,
    confidence,
    emotion,

    # High-level
    trustworthiness,
    hireability,
    mental_resilience,
    social_dominance,
    psychological_fatigue,
    adaptability
):

    return [
        # Timestamp
        timestamp,
    
        # Mouth
        mouth_open_raw,
        mouth_change,
        mouth_activity,
        mouth_smoothed,
        mouth_final,
    
        # Chest
        chest_flow_raw,
        chest_activity,
        chest_final,
    
        # Eyes
        left_eye_intensity,
        right_eye_intensity,
        eye_intensity_mean,
        eye_activity,
        eye_final,
    
        # Gaze
        gaze_center_x,
        gaze_center_y,
        gaze_label,
        gaze_penalty,
        gaze_final,
    
        # Audio
        audio_energy,
        audio_pitch,
        audio_zero_cross_rate,
        audio_peak,
        audio_energy_normalized,
        audio_final,
    
        # Fusion inputs
        mouth_for_fusion,
        energy_normalized,
        gaze_penalty,
    
        # Final states
        stress,
        focus,
        confidence,
        emotion,
    
        # High-level interview metrics
        trustworthiness,
        hireability,
        mental_resilience,
        social_dominance,
        psychological_fatigue,
        adaptability
    ]


# ============================================================
# MAIN LOOP + SUMMARY
# ============================================================
def draw_text_with_bg(img, text, pos,
                      font=cv2.FONT_HERSHEY_SIMPLEX,
                      scale=0.5,
                      text_color=(0, 255, 255),   # yellow
                      bg_color=(0, 0, 0),         # black
                      thickness=1,
                      padding=6):
    x, y = pos
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)

    # black rectangle exactly behind this line (not full width)
    cv2.rectangle(
        img,
        (x - padding, y - h - padding),
        (x + w + padding, y + padding),
        bg_color,
        -1
    )

    cv2.putText(img, text, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)
    
def put_text_with_bg(img, text, org, font, scale, color, thickness):
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    # black background
    cv2.rectangle(
        img,
        (x - 5, y - h - 5),
        (x + w + 5, y + baseline + 5),
        (0, 0, 0),
        -1
    )
    cv2.putText(
        img, text, org, font, scale, color, thickness, cv2.LINE_AA
    )

def draw_persian_overlay(frame, overlay_lines):
    """
    overlay_lines: list of tuples (text, is_big)
    text: Persian string
    is_big: True/False for size
    """

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    y = overlay_lines[0][2] if len(overlay_lines[0]) == 3 else 30

    x = 15

    for item in overlay_lines:
        text = item[0]
        is_big = item[1]

        reshaped = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped)

        font_size = 22 if is_big else 18
        font = ImageFont.truetype(PERSIAN_FONT_PATH, font_size)

        if is_big:
            color = (255, 0, 0)
        else:
            color = (255, 255, 0)

        draw.text(
            (x, y),
            bidi_text,
            font=font,
            fill=color,
            stroke_width=2 if is_big else 1,
            stroke_fill=(0, 0, 0)
        )

        y += 30 if is_big else 18

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def main():
    # Init outputs
    init_all_output_files()

    print("SYSTEM STARTED — PRESS Q OR ESC TO STOP")

    # Audio
    audio_stream = start_audio_stream()

    # Camera
# Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Camera could not be opened")
        audio_stream.stop()
        audio_stream.close()
        sd.stop()
        return


    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    prev_gray = None
    prev_mouth_distance = None
    smoothed_mouth = 0.0
    frame_index = 0

    # Stats for summary
    all_stress = []
    all_focus = []
    all_confidence = []
    all_emotions = []

    all_trustworthiness = []
    all_hireability = []
    all_resilience = []
    all_dominance = []
    all_fatigue = []
    all_adaptability = []

    fps_last_time = time.time()
    fps_count = 0
    fps_display = 0.0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("WARNING: Camera frame not available")
            time.sleep(0.05)
            continue


        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        now = time.time()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))

        face_rect = get_main_face(gray)

        # Defaults
        mouth_open_raw = 0.0
        mouth_change = 0.0
        mouth_activity = 0.0
        mouth_box = None

        chest_flow_raw = 0.0
        chest_activity = 0.0
        chest_box = None

        left_eye_intensity = 0.0
        right_eye_intensity = 0.0
        eye_intensity_mean = 0.0
        eye_activity = 0.0
        gaze_center_x = 0.5
        gaze_center_y = 0.5
        gaze_label = "Center"

        if face_rect is not None:
            # Mouth
            (
                mouth_open_raw,
                mouth_change,
                mouth_activity,
                mouth_box,
                prev_mouth_distance
            ) = extract_mouth_features(
                gray, face_rect, prev_mouth_distance
            )

            # Chest
            (
                chest_flow_raw,
                chest_activity,
                chest_box
            ) = extract_chest_features(prev_gray, gray, face_rect)

            # Eyes + Gaze
            (
                left_eye_intensity,
                right_eye_intensity,
                eye_intensity_mean,
                eye_activity,
                gaze_center_x,
                gaze_center_y,
                gaze_label,
                eye_boxes
            ) = extract_eye_features(gray, face_rect)
            


            # Draw face rectangle
            x, y, w, h = face_rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Audio
        (
            audio_energy,
            audio_pitch,
            audio_zero_cross_rate,
            audio_rms,
            audio_peak
        ) = compute_audio_features()

        # Mouth gating by audio
        # Mouth gating by audio (soft gate)
        gate = np.clip((audio_energy - AUDIO_GATE_THRESHOLD) / AUDIO_GATE_THRESHOLD, 0.0, 1.0)
        mouth_for_fusion = mouth_activity * (0.3 + 0.7 * gate)



        # Smooth mouth
        smoothed_mouth = (
            MOUTH_SMOOTH_ALPHA * smoothed_mouth +
            (1.0 - MOUTH_SMOOTH_ALPHA) * mouth_for_fusion
        )

        # Fusion states
        (
            energy_normalized,
            gaze_penalty,
            stress,
            focus,
            confidence,
            emotion
        ) = compute_fusion_states(
            smoothed_mouth,
            chest_activity,
            eye_activity,
            audio_energy,
            gaze_label
        )

        # High-level metrics
        (
            trustworthiness,
            hireability,
            mental_resilience,
            social_dominance,
            psychological_fatigue,
            adaptability
        ) = compute_high_level_metrics(
            stress,
            focus,
            confidence,
            eye_activity,
            chest_activity,
            energy_normalized
        )
        mouth_final = smoothed_mouth
        chest_final = chest_activity
        eye_final = eye_activity
        audio_final = energy_normalized
        gaze_final = gaze_penalty
        
        # ---- Emotion English → Persian (logic unchanged) ----
        emotion_map = {
            "Anxious": "مضطرب",
            "Stressed": "پراسترس",
            "Confident": "با اعتماد به نفس",
            "Focused": "متمرکز",
            "Bored": "بی‌حوصله",
            "Engaged": "درگیر",
            "Distracted": "حواس‌پرت",
            "Neutral": "خنثی"
        }
        
        emotion_fa = emotion_map.get(emotion, emotion)


        # Store for summary
        all_stress.append(stress)
        all_focus.append(focus)
        all_confidence.append(confidence)
        all_emotions.append(emotion)

        all_trustworthiness.append(trustworthiness)
        all_hireability.append(hireability)
        all_resilience.append(mental_resilience)
        all_dominance.append(social_dominance)
        all_fatigue.append(psychological_fatigue)
        all_adaptability.append(adaptability)

        # Build rows
        combined_row = assemble_combined_row(
            timestamp,
        
            # Mouth
            mouth_open_raw,
            mouth_change,
            mouth_activity,
            smoothed_mouth,
            mouth_final,
        
            # Chest
            chest_flow_raw,
            chest_activity,
            chest_final,
        
            # Eyes
            left_eye_intensity,
            right_eye_intensity,
            eye_intensity_mean,
            eye_activity,
            eye_final,
        
            # Gaze
            gaze_center_x,
            gaze_center_y,
            gaze_label,
            gaze_penalty,
            gaze_final,
        
            # Audio
            audio_energy,
            audio_pitch,
            audio_zero_cross_rate,
            audio_peak,
            energy_normalized,
            audio_final,
        
            # Fusion inputs
            mouth_for_fusion,
            energy_normalized,
            # gaze_penalty,
        
            # Final states
            stress,
            focus,
            confidence,
            emotion,
        
            # High-level metrics
            trustworthiness,
            hireability,
            mental_resilience,
            social_dominance,
            psychological_fatigue,
            adaptability
        )

        reduced_row = [
            timestamp,
            smoothed_mouth,
            chest_activity,
            eye_activity,
            gaze_label,
            audio_energy,
            audio_pitch,
            stress,
            focus,
            confidence,
            emotion,
            trustworthiness,
            hireability,
            mental_resilience,
            social_dominance,
            psychological_fatigue,
            adaptability
        ]

        # Save combined / reduced with sampling
        if frame_index % SAVE_RATE == 0:
            append_txt(COMBINED_TXT, combined_row)
            append_csv(COMBINED_CSV, combined_row)
            append_tsv(COMBINED_TSV, combined_row)

        if frame_index % SAVE_REDUCED_RATE == 0:
            append_txt(REDUCED_TXT, reduced_row)
            append_csv(REDUCED_CSV, reduced_row)
            append_tsv(REDUCED_TSV, reduced_row)

        # FPS calculation
        fps_count += 1
        if now - fps_last_time >= 1.0:
            fps_display = fps_count / (now - fps_last_time)
            fps_last_time = now
            fps_count = 0

    
        # ---- Persian Time + Emotion ----
# ---- FULL PERSIAN OVERLAY (FINAL) ----
        frame = draw_persian_overlay(
            frame,
            [
                (f"زمان: {timestamp}", False, 30),
                (f"فریم‌برثانیه: {fps_display:.1f}", False, 50),
        
                (f"حرکت دهان: {smoothed_mouth:.2f}", False, 70),
                (f"حرکت قفسه سینه: {chest_activity:.2f}", False, 90),
                (f"فعالیت چشم: {eye_activity:.2f}", False, 110),
                (f"جهت نگاه: {gaze_label}", False, 130),
        
                (f"انرژی صدا: {audio_energy:.3f}", False, 150),
                (f"فرکانس صدا: {audio_pitch:.1f} هرتز", False, 170),
        
                (f"استرس: {stress:.2f}", True, 200),
                (f"تمرکز: {focus:.2f}", True, 235),
                (f"اعتمادبه‌نفس: {confidence:.2f}", True, 270),
        
                (f"احساس: {emotion_fa}", True, 305),
        
                (f"قابل‌اعتماد بودن: {trustworthiness:.2f}", False, 340),
                (f"قابلیت استخدام: {hireability:.2f}", False, 360),
                (f"تاب‌آوری ذهنی: {mental_resilience:.2f}", False, 380),
                (f"غلبه اجتماعی: {social_dominance:.2f}", False, 400),
                (f"خستگی روانی: {psychological_fatigue:.2f}", False, 420),
                (f"سازگاری: {adaptability:.2f}", False, 440),
            ]
        )


        # Mouth box (BLUE)
        if mouth_box is not None:
            mx, my, mw, mh = mouth_box
            cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (255, 0, 0), 2)
            # Eye boxes (YELLOW)
            # Eye boxes (YELLOW) — FIXED COORDINATES
            fx, fy, fw, fh = face_rect
            eye_y_offset = int(fh * 0.5)
            
            for (ex, ey, ew, eh) in eye_boxes:
                x1 = fx + ex
                y1 = fy + ey
                x2 = x1 + ew
                y2 = y1 + eh
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)


        
        # Chest box (MAGENTA)
        if chest_box is not None:
            cx, cy, cw, ch = chest_box
            cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), (255, 0, 255), 2)
        
        frame_big = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        frame_index += 1

        cv2.imshow("Interview Monitor - FULL", frame_big)
        prev_gray = gray

        key = cv2.waitKey(1) & 0xFF
        if key in [27, ord('q')]:
            break


    # ================= FINAL SUMMARY =================

    # cap.release()
    # cv2.destroyAllWindows()
    # audio_stream.stop()
    # audio_stream.close()
    # sd.stop()


    print("SYSTEM STOPPED — COMPUTING FINAL SUMMARY")

    mean_stress = float(np.mean(all_stress)) if all_stress else 0.0
    mean_focus = float(np.mean(all_focus)) if all_focus else 0.0
    mean_conf = float(np.mean(all_confidence)) if all_confidence else 0.0

    counts = Counter(all_emotions)
    if counts:
        total = float(sum(counts.values()))
        emotion_percent = {k: round(100.0 * v / total, 2) for k, v in counts.items()}
        dominant_emotion = max(emotion_percent, key=emotion_percent.get)
    else:
        emotion_percent = {}
        dominant_emotion = "None"

    avg_trust = float(np.mean(all_trustworthiness)) if all_trustworthiness else 0.0
    avg_hire = float(np.mean(all_hireability)) if all_hireability else 0.0
    avg_res = float(np.mean(all_resilience)) if all_resilience else 0.0
    avg_dom = float(np.mean(all_dominance)) if all_dominance else 0.0
    avg_fat = float(np.mean(all_fatigue)) if all_fatigue else 0.0
    avg_adapt = float(np.mean(all_adaptability)) if all_adaptability else 0.0

    summary_rows = [
        ["Mean Stress", mean_stress],
        ["Mean Focus", mean_focus],
        ["Mean Confidence", mean_conf],
        ["Dominant Emotion", dominant_emotion],
        ["Emotion Distribution", emotion_percent],
        ["Avg Trustworthiness", avg_trust],
        ["Avg Hireability", avg_hire],
        ["Avg Mental Resilience", avg_res],
        ["Avg Social Dominance", avg_dom],
        ["Avg Psychological Fatigue", avg_fat],
        ["Avg Adaptability", avg_adapt]
    ]

    with open(SUMMARY_TXT, "a", encoding="utf-8") as f:
        for m, v in summary_rows:
            f.write(f"{m}: {v}\n")

    with open(SUMMARY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for m, v in summary_rows:
            writer.writerow([m, v])

    with open(SUMMARY_TSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        for m, v in summary_rows:
            writer.writerow([m, v])

    print("FINAL SUMMARY SAVED (TXT, CSV, TSV)")
    # ================= CLEAN EXIT =================
    cap.release()
    cv2.destroyAllWindows()
    
    audio_stream.stop()
    audio_stream.close()
    sd.stop()

    print("DOMINANT EMOTION:", dominant_emotion)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
