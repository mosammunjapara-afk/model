from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import gc

# Configure TensorFlow for CPU efficiency
tf.config.set_soft_device_placement(True)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

MODEL_PATH = "fruit_model.keras"
model = None
IMG_SIZE = 224  # ✅ MUST BE 224 (matches training!)

def load_model_safe():
    """Load model with compatibility fixes"""
    global model
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        return False
    
    try:
        print("📥 Attempting to load model...")
        model = keras.models.load_model(MODEL_PATH, compile=False)
        
        # Compile with simpler optimizer
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Warmup prediction
        print("🔥 Warming up model...")
        dummy_input = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        _ = model.predict(dummy_input, verbose=0)
        
        print("✅ Model loaded and warmed up successfully!")
        print(f"📊 Model size: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")
        
        gc.collect()
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

# Load model on startup
print("🚀 Starting application...")
if not load_model_safe():
    print("⚠️ WARNING: Running without model!")

class_names = [
    "Angelina Jolie", "Brad Pitt", "Denzel Washington", "Hugh Jackman",
    "Jennifer Lawrence", "Johnny Depp", "Kate Winslet", "Leonardo DiCaprio",
    "Megan Fox", "Natalie Portman", "Nicole Kidman", "Robert Downey Jr",
    "Sandra Bullock", "Scarlett Johansson", "Tom Cruise", "Tom Hanks", "Will Smith"
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
            error = "⚠️ Model not loaded. Service is starting up, please wait 30 seconds and try again."
            return render_template("index.html", prediction=None, confidence=None, 
                                 img_path=None, error=error)
        
        file = request.files.get("image")

        if file and file.filename:
            try:
                print(f"📁 Received: {file.filename}")
                
                img_path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(img_path)
                print("💾 File saved")

                print(f"🖼️ Loading image at size {IMG_SIZE}x{IMG_SIZE}...")
                img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
                img_array = image.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
                print(f"✅ Image shape: {img_array.shape}")

                print("🔍 Predicting...")
                import time
                start = time.time()
                
                # Optimized prediction
                preds = model.predict(img_array, verbose=0, batch_size=1)
                
                elapsed = time.time() - start
                print(f"✅ Prediction done in {elapsed:.2f}s")
                
                max_prob = float(np.max(preds)) * 100
                class_index = np.argmax(preds)

                if max_prob < CONFIDENCE_THRESHOLD:
                    prediction = "Unknown Person ⚠️"
                    confidence = round(max_prob, 2)
                else:
                    prediction = class_names[class_index]
                    confidence = round(max_prob, 2)
                
                print(f"🎯 Result: {prediction} ({confidence}%)")
                
                # Cleanup
                del img_array, preds
                gc.collect()
                
            except Exception as e:
                error = f"Error processing image: {str(e)}"
                print(f"❌ Error: {error}")
        else:
            error = "Please select an image file."

    return render_template("index.html", prediction=prediction, confidence=confidence,
                         img_path=img_path, error=error)

@app.route("/health")
def health():
    """Health check endpoint"""
    status = "healthy" if model is not None else "model_not_loaded"
    return {"status": status, "model_loaded": model is not None}, 200 if model else 503

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🌐 Starting Flask on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
