from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import gdown
import sys

app = Flask(__name__)

MODEL_PATH = "fruit_model.h5"
MODEL_URL = "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID_HERE"   # ← CHANGE THIS

# Download model only if not already present
if not os.path.exists(MODEL_PATH):
    print(f"Downloading model from Google Drive...", file=sys.stderr)
    try:
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)
        print("Model downloaded successfully", file=sys.stderr)
    except Exception as e:
        print(f"ERROR downloading model: {e}", file=sys.stderr)
        raise

print("Loading model...", file=sys.stderr)
model = load_model(MODEL_PATH, compile=False)
print("Model loaded", file=sys.stderr)

# ──────────────────────────────────────────────
# rest is same as above
class_names = [ ... ]   # your list

CONFIDENCE_THRESHOLD = 70

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    # ... same code as in Option 1 ...
    pass   # ← copy prediction logic from above

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
