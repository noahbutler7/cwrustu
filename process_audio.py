from flask import Flask, request, jsonify, send_from_directory
import os
import librosa
import numpy as np
import soundfile as sf
import subprocess

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/outputs/<path:filename>')
def serve_audio(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

def separate_stems(input_path, output_dir):
    command = f"demucs -o {output_dir} {input_path}"
    subprocess.run(command, shell=True, check=True)
    song_name = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(output_dir, "htdemucs", song_name)

def detect_notes(audio_path, sr=22050):
    y, _ = librosa.load(audio_path, sr=sr)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    note_map = {}
    seen = set()
    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch = pitches[index, i]
        if pitch > 0:
            note = librosa.hz_to_note(pitch)[0]
            if note not in seen and note in "ABCDEFG":
                seen.add(note)
                note_map[note] = i * 512 / sr
        if len(note_map) >= 7:
            break
    return note_map

def extract_and_save(audio_path, note_map, output_dir, sr=22050):
    y, _ = librosa.load(audio_path, sr=sr)
    os.makedirs(output_dir, exist_ok=True)
    audio_files = {}
    for note, start_time in note_map.items():
        start = int(start_time * sr)
        end = start + int(sr * 1.0)
        slice_audio = y[start:end]
        filename = f"{note}.wav"
        sf.write(os.path.join(output_dir, filename), slice_audio, sr)
        audio_files[note] = f"{output_dir}/{filename}"
    return audio_files

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    stem_dir = separate_stems(filepath, OUTPUT_FOLDER)
    audio_response = {}

    for stem in os.listdir(stem_dir):
        stem_path = os.path.join(stem_dir, stem)
        stem_name = os.path.splitext(stem)[0]
        note_map = detect_notes(stem_path)
        output_stem_dir = os.path.join(OUTPUT_FOLDER, stem_name)
        audio_files = extract_and_save(stem_path, note_map, output_stem_dir)
        audio_response[stem_name] = audio_files

    return jsonify({"audio_files": audio_response})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
