from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# ==================================================
# SOLUTION 1: Google Drive (Recommended for large files)
# ==================================================
# Replace YOUR_FILE_ID with your actual Google Drive file ID
# Instructions below in DEPLOYMENT_GUIDE.md

MODEL_PATH = "fruit_model.keras"
model = None

def download_model_from_gdrive():
    """Download model from Google Drive if not present"""
    GOOGLE_DRIVE_FILE_ID = os.environ.get("GDRIVE_FILE_ID", "YOUR_FILE_ID")
    
    if GOOGLE_DRIVE_FILE_ID == "YOUR_FILE_ID":
        print("⚠️ WARNING: Google Drive File ID not set!")
        print("Set environment variable GDRIVE_FILE_ID in Render dashboard")
        return False
    
    if os.path.exists(MODEL_PATH):
        print("✅ Model file already exists")
        return True
    
    try:
        import gdown
        MODEL_URL = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        print(f"📥 Downloading model from Google Drive...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        print("✅ Model downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False

def load_model_safe():
    """Load model with error handling"""
    global model
    
    # Try to download if not exists
    if not os.path.exists(MODEL_PATH):
        print("Model file not found, attempting download...")
        download_model_from_gdrive()
    
    # Try to load model
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            print("✅ Model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    else:
        print("❌ Model file not found and download failed")
        return False

# Load model on startup (won't crash if fails)
load_model_safe()

class_names = [
    "Angelina Jolie",
    "Brad Pitt",
    "Denzel Washington",
    "Hugh Jackman",
    "Jennifer Lawrence",
    "Johnny Depp",
    "Kate Winslet",
    "Leonardo DiCaprio",
    "Megan Fox",
    "Natalie Portman",
    "Nicole Kidman",
    "Robert Downey Jr",
    "Sandra Bullock",
    "Scarlett Johansson",
    "Tom Cruise",
    "Tom Hanks",
    "Will Smith"
]

CONFIDENCE_THRESHOLD = 70

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    img_path = None
    error = None

    if request.method == "POST":
        if model is None:
            error = "⚠️ Model not loaded. Please contact administrator."
            return render_template(
                "index.html",
                prediction=None,
                confidence=None,
                img_path=None,
                error=error
            )
        
        file = request.files.get("image")

        if file and file.filename:
            try:
                img_path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(img_path)

                img = image.load_img(img_path, target_size=(224, 224))
                img_array = image.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                preds = model.predict(img_array)
                max_prob = float(np.max(preds)) * 100
                class_index = np.argmax(preds)

                if max_prob < CONFIDENCE_THRESHOLD:
                    prediction = "Unknown  ⚠️"
                    confidence = round(max_prob, 2)
                else:
                    prediction = class_names[class_index]
                    confidence = round(max_prob, 2)
            except Exception as e:
                error = f"Error processing image: {str(e)}"
                print(f"Error: {error}")

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        img_path=img_path,
        error=error
    )

@app.route("/health")
def health():
    """Health check endpoint"""
    status = "healthy" if model is not None else "model_not_loaded"
    return {"status": status}, 200 if model else 503

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
