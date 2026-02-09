from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = "fruit_model.keras"
model = None

def load_model_safe():
    """Load model with compatibility fixes"""
    global model
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        return False
    
    try:
        # Try loading with compile=False to avoid optimizer issues
        print("📥 Attempting to load model...")
        model = keras.models.load_model(MODEL_PATH, compile=False)
        
        # Recompile if needed
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model loaded and compiled successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Trying alternative loading method...")
        
        try:
            # Alternative: Load with custom objects if needed
            model = keras.models.load_model(
                MODEL_PATH,
                compile=False,
                safe_mode=False
            )
            
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print("✅ Model loaded with alternative method!")
            return True
            
        except Exception as e2:
            print(f"❌ Alternative method also failed: {e2}")
            return False

# Load model on startup
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
            error = "⚠️ Model not loaded. Please check server logs."
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

                preds = model.predict(img_array, verbose=0)
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
