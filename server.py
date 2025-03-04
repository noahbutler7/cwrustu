from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import librosa
from mido import Message, MidiFile, MidiTrack
import demucs.separate

app = Flask(__name__)
CORS(app) # Enable CORS everywhere (not safe)
SEPARATION_MODEL = "htdemucs"
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = f"separated/{SEPARATION_MODEL}" # Subdirectory changes based on model used
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Function to extract MIDI notes
def extract_notes_from_audio(audio_file, sr=22050):
    y, sr = librosa.load(audio_file, sr=sr)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    
    midi_notes = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            midi_note = int(librosa.hz_to_midi(pitch))
            midi_notes.append(midi_note)

    return midi_notes

# Function to create a MIDI file
def create_midi_from_notes(midi_notes, output_midi_file="output.mid"):
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)
    
    track.append(Message('program_change', program=0, time=0))
    for note in midi_notes:
        track.append(Message('note_on', note=note, velocity=64, time=0))
        track.append(Message('note_off', note=note, velocity=64, time=200))

    midi.save(output_midi_file)
    return output_midi_file

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith(".mp3"):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Separate stems
        output_dir = os.path.join(OUTPUT_FOLDER, filename.split(".")[0])
        #os.makedirs(output_dir, exist_ok=True)
        demucs.separate.main(["--mp3", file_path]) # See alternate models: https://github.com/facebookresearch/demucs?tab=readme-ov-file#separating-tracks
        # alternate model call (fine-tuned model, takes 4x longer): demucs.separate.main(["--mp3", "-n", "htdemucs_ft", file_path])

        # Extract MIDI notes from the melody track
        melody_path = os.path.join(output_dir, "other.mp3")  # Spleeter's melody track
        midi_notes = extract_notes_from_audio(melody_path)
        midi_file = os.path.join(OUTPUT_FOLDER, f"{filename.split('.')[0]}.mid")
        create_midi_from_notes(midi_notes, midi_file)

        return jsonify({"message": "Processing complete", "midi_file": midi_file}), 200

    return jsonify({"error": "Invalid file format"}), 400

@app.route("/download/<filename>")
def download_file(filename):
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({"error": "File not found"}), 404

if __name__ == "__main__":
    app.run(debug=True, port=5000)