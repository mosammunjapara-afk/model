from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import gc

# Configure TensorFlow for memory efficiency
tf.config.set_soft_device_placement(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging

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
        
        # Recompile with lower memory optimizer
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model loaded and compiled successfully!")
        print(f"📊 Model size: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")
        
        # Clear memory
        gc.collect()
        
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
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print("✅ Model loaded with alternative method!")
            gc.collect()
            return True
            
        except Exception as e2:
            print(f"❌ Alternative method also failed: {e2}")
            return False

# Load model on startup
print("🚀 Starting application...")
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
            error = "⚠️ Model not loaded. Please check server logs or restart the service."
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
                # Save uploaded file
                img_path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(img_path)

                # Load and preprocess image
                img = image.load_img(img_path, target_size=(224, 224))
                img_array = image.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Predict with optimized settings
                print("🔍 Making prediction...")
                preds = model.predict(img_array, verbose=0, batch_size=1)
                
                max_prob = float(np.max(preds)) * 100
                class_index = np.argmax(preds)

                if max_prob < CONFIDENCE_THRESHOLD:
                    prediction = "Unknown Person ⚠️"
                    confidence = round(max_prob, 2)
                else:
                    prediction = class_names[class_index]
                    confidence = round(max_prob, 2)
                
                print(f"✅ Prediction: {prediction} ({confidence}%)")
                
                # Clean up memory
                del img_array, preds
                gc.collect()
                
            except Exception as e:
                error = f"Error processing image: {str(e)}"
                print(f"❌ Error: {error}")
        else:
            error = "Please select an image file to upload."

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
    model_loaded = model is not None
    return {
        "status": status,
        "model_loaded": model_loaded
    }, 200 if model else 503

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🌐 Starting Flask on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
