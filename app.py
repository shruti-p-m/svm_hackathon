import os
import tempfile
import whisper
import ffmpeg
from flask import Flask, request, render_template, send_from_directory
from pydub import AudioSegment
from transformers import pipeline
from datetime import timedelta
import re
import spacy
import whisperx
import torch
from pyannote.audio import Pipeline
import numpy as np
from scipy.io.wavfile import write as write_wav
from config_secret import API_TOKEN

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models
whisper_model = whisper.load_model("base")
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)
summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
nlp = spacy.load("en_core_web_sm")

# Helpers
def format_timestamp(seconds):
    return str(timedelta(seconds=seconds)).split(".")[0] + ",000"

def convert_to_mp3(input_file):
    output_file = os.path.join(UPLOAD_FOLDER, "converted_audio.mp3")
    (
        ffmpeg
        .input(input_file)
        .output(output_file, format="mp3", acodec="libmp3lame")
        .overwrite_output()
        .run()
    )
    return output_file

def split_text(text, max_len=1000):
    sentences = text.split(". ")
    chunks, chunk = [], ""
    for sentence in sentences:
        if len(chunk) + len(sentence) < max_len:
            chunk += sentence + ". "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + ". "
    if chunk:
        chunks.append(chunk.strip())
    return chunks

def generate_meeting_minutes(transcript, summary):
    def extract_purpose(summary):
        for line in summary.split("."):
            if any(keyword in line.lower() for keyword in ["purpose", "intent", "reason", "objective"]):
                return line.strip().capitalize() + "."
        return "The purpose of the meeting was to discuss relevant topics based on the conversation."

    def extract_participants(text):
        doc = nlp(text)
        names = {ent.text.split()[0] for ent in doc.ents if ent.label_ == "PERSON"}
        filtered = [name for name in names if name.lower() not in {"monday", "friday", "gym", "yoga", "class"}]
        return ", ".join(sorted(filtered)) if filtered else "Not clearly mentioned."

    def extract_key_points(summary):
        sentences = summary.strip().split(". ")
        return "- " + "\n- ".join(sentences[:3]) + "."

    def extract_decisions(summary):
        for line in summary.split("."):
            if any(word in line.lower() for word in ["decided", "chose", "agreed", "selected"]):
                return line.strip().capitalize() + "."
        return "Relevant decisions were made during the conversation."

    def extract_action_items(summary):
        for line in summary.split("."):
            if any(word in line.lower() for word in ["will", "needs to", "should", "expected to"]):
                return line.strip().capitalize() + "."
        return "Participants were expected to take appropriate next steps."

    purpose = extract_purpose(summary)
    participants = extract_participants(transcript)
    key_points = extract_key_points(summary)
    decisions = extract_decisions(summary)
    actions = extract_action_items(summary)

    return f"""
Meeting Minutes

📅 Meeting Purpose:
•⁠  ⁠{purpose}

👥 Participants:
•⁠  ⁠{participants}

📜 Key Discussion Points:
{key_points}

✅ Decisions Made:
•⁠  ⁠{decisions}

📌 Action Items:
•⁠  ⁠{actions}

📄 Summary:
{summary}
"""

def srt_to_vtt(srt_text):
    vtt_text = "WEBVTT\n\n"
    blocks = srt_text.strip().split("\n\n")
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) >= 3:
            index = lines[0]
            timestamp = lines[1].replace(",", ".")
            text = "\n".join(lines[2:])
            vtt_text += f"{timestamp}\n{text}\n\n"
    return vtt_text

# Helper function to format timestamp as HH:MM:SS
def format_timestamp(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

# Initialize WhisperX and diarization model
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisperx.load_model("large-v2", device=device, compute_type="float32")

# Diarization model
os.environ['HF_TOKEN'] = API_TOKEN
diarize_model = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=os.environ["HF_TOKEN"])

# --- HELPER ---
def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    return str(td)[:-3] if "." in str(td) else str(td) + ".000"

def simple_assign_speakers(diarized_segments, aligned_segments):
    output_segments = []
    for word_segment in aligned_segments:
        word_start = word_segment["start"]
        word_end = word_segment["end"]
        speaker_label = "Unknown"

        max_overlap = 0.0
        best_speaker = None

        for turn in diarized_segments.itertracks(yield_label=True):
            (segment, _, speaker) = turn
            overlap_start = max(word_start, segment.start)
            overlap_end = min(word_end, segment.end)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = speaker

        if best_speaker:
            speaker_label = best_speaker

        word_segment["speaker"] = speaker_label
        output_segments.append(word_segment)

    return {"segments": output_segments}

# --- MAIN FUNCTION ---
def process_audio_with_diarization(audio_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ['HF_TOKEN'] = API_TOKEN

    # Load audio and model
    audio = whisperx.load_audio(audio_path)
    model = whisperx.load_model("large-v2", device=device, compute_type="float32")

    # Transcribe and translate to English
    transcription_result = model.transcribe(audio_path, task="translate")
    detected_language = transcription_result.get("language", "unknown")

    segments = transcription_result.get("segments", [])
    text = " ".join([seg["text"].strip() for seg in segments])

    # Load alignment model for original language (required even if translated)
    model_a, metadata = whisperx.load_align_model(language_code=detected_language, device=device)
    aligned = whisperx.align(transcription_result["segments"], model_a, metadata, audio, device)

    # Diarize
    diarized_segments = diarize_model({
        "uri": "conversation_sample",
        "audio": audio_path
    })

    # Assign speakers
    final_output = simple_assign_speakers(diarized_segments, aligned["segments"])

    # Initialize output containers
    caption_output = ""
    sentiment_data = []
    emotion_data = []
    raw_transcript = ""
    srt_index = 1

    for segment in final_output["segments"]:
        start_sec = int(segment["start"])
        end_sec = int(segment["end"])
        text = segment["text"].strip()
        speaker = segment["speaker"]

        sentiment = sentiment_model(text)[0]
        emotion = emotion_model(text)[0][0]

        start_ts = format_timestamp(start_sec)
        end_ts = format_timestamp(end_sec)

        caption_output += f"{srt_index}\n{start_ts} --> {end_ts}\n{speaker}: {text}\n\n"
        srt_index += 1
        raw_transcript += f"{speaker}: {text}\n"

        sentiment_data.append({
            "speaker": speaker,
            "label": sentiment["label"],
            "score": sentiment["score"]
        })

        emotion_data.append({
            "speaker": speaker,
            "label": emotion["label"],
            "score": emotion["score"]
        })

    def get_top_label(data):
        df = {}
        for entry in data:
            sp = entry["speaker"]
            label = entry["label"]
            score = entry["score"]
            df.setdefault(sp, {}).setdefault(label, []).append(score)
        top_results = {}
        for sp, label_scores in df.items():
            top_label = max(label_scores.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))
            top_results[sp] = {
                "label": top_label[0],
                "score": round(sum(top_label[1]) / len(top_label[1]), 3)
            }
        return top_results

    top_sentiment = get_top_label(sentiment_data)
    top_emotion = get_top_label(emotion_data)

    print(f"Sentiment Data: {sentiment_data}")
    print(f"Emotion Data: {emotion_data}")
    print(f"Top Sentiment: {top_sentiment}")
    print(f"Top Emotion: {top_emotion}")

    return caption_output.strip(), top_sentiment, top_emotion, raw_transcript


@app.route("/", methods=["GET", "POST"])
def index():
    caption_text = sentiment_json = emotion_json = meeting_minutes = None
    video_filename = caption_file = None
    top_sentiment = top_emotion = {}

    if request.method == "POST":
        file = request.files.get("file")
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            ext = os.path.splitext(file_path)[1].lower()
            if ext in [".mp4", ".mov", ".avi"]:
                audio_path = convert_to_mp3(file_path)
                video_filename = file.filename
            else:
                audio_path = file_path

            caption_text, top_sentiment, top_emotion, raw_transcript = process_audio_with_diarization(audio_path)
            chunks = split_text(raw_transcript)
            summaries = [summarizer(chunk, max_length=120, min_length=30, do_sample=False)[0]["summary_text"] for chunk in chunks]
            summary_text = " ".join(summaries)
            meeting_minutes = generate_meeting_minutes(raw_transcript, summary_text)

            vtt_text = srt_to_vtt(caption_text)
            caption_file = "captions.vtt"
            with open(os.path.join(UPLOAD_FOLDER, caption_file), "w", encoding="utf-8") as f:
                f.write(vtt_text)

    return render_template("index.html", 
                           caption_text=caption_text, 
                           top_sentiment=top_sentiment,
                           top_emotion=top_emotion,
                           meeting_minutes=meeting_minutes,
                           video_filename=video_filename,
                           caption_file=caption_file)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)