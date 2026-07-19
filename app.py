from difflib import SequenceMatcher
from flask import Flask, render_template, request, send_file
import os
import subprocess
import librosa
import torch
import csv
from datetime import datetime

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration
)

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
CSV_FILE = "student_results.csv"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Loading Fine-Tuned Model...")

processor = WhisperProcessor.from_pretrained(
    "./whisper-assamese"
)

model = WhisperForConditionalGeneration.from_pretrained(
    "./whisper-assamese"
)

print("Model Loaded!")


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/analyzer")
def analyzer():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():

    data = []

    if os.path.exists(CSV_FILE):

        with open(
            CSV_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            reader = csv.reader(file)

            next(reader, None)

            for row in reader:

                if len(row) > 0:
                    data.append(row)

    total_students = len(data)

    average_accuracy = round(
        sum(float(row[1]) for row in data) / total_students,
        2
    ) if total_students > 0 else 0

    excellent_count = sum(
        1 for row in data
        if row[2] == "Excellent"
    )

    highest_accuracy = max(
        [float(row[1]) for row in data],
        default=0
    )

    return render_template(
        "dashboard.html",
        data=data,
        total_students=total_students,
        average_accuracy=average_accuracy,
        excellent_count=excellent_count,
        highest_accuracy=highest_accuracy
    )

from flask import send_file


@app.route("/download")
def download():

    return send_file(
        CSV_FILE,
        as_attachment=True
    )

@app.route("/upload", methods=["POST"])
def upload():

    student_name = request.form["student_name"]

    expected_text = request.form["expected_text"]

    audio = request.files["audio"]

    webm_path = os.path.join(
        UPLOAD_FOLDER,
        "student_recording.webm"
    )

    wav_path = os.path.join(
        UPLOAD_FOLDER,
        "student_recording.wav"
    )

    audio.save(webm_path)

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            webm_path,
            wav_path
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    audio_data, sr = librosa.load(
        wav_path,
        sr=16000
    )

    inputs = processor(
        audio_data,
        sampling_rate=16000,
        return_tensors="pt"
    )

    with torch.no_grad():

        predicted_ids = model.generate(
            inputs.input_features,
            language="as",
            task="transcribe"
        )

    text = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]

    accuracy = round(
        SequenceMatcher(
            None,
            expected_text.lower(),
            text.lower()
        ).ratio() * 100,
        2
    )

    if accuracy >= 90:
        rating = "Excellent"

    elif accuracy >= 75:
        rating = "Good"

    elif accuracy >= 60:
        rating = "Average"

    else:
        rating = "Needs Practice"

    result = f"""
Student Name:
{student_name}

Expected Text:
{expected_text}

Recognized Text:
{text}

Accuracy:
{accuracy}%

Fluency Rating:
{rating}
"""
    current_time = datetime.now().strftime(
        "%d-%m-%Y %H:%M"
    )

    file_exists = os.path.isfile(CSV_FILE)

    with open(
        CSV_FILE,
        mode="a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Student Name",
                "Accuracy",
                "Rating",
                "Date"
            ])

        writer.writerow([
            student_name,
            accuracy,
            rating,
            current_time
        ])

    return result


if __name__ == "__main__":
    app.run(
        debug=True,
        port=5001
    )