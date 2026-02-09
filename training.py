import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow.keras.mixed_precision as mixed_precision

# Enable mixed precision for faster training
mixed_precision.set_global_policy('mixed_float16')

IMG_SIZE = 160  # Reduced from 224 - IMPORTANT!
BATCH_SIZE = 32
EPOCHS = 15

train_dir = "dataset/train"
test_dir = "dataset/test"

# Data augmentation
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.15,
    height_shift_range=0.15
)

test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# Load LIGHTER pretrained model
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
    alpha=0.5  # Use 0.5 width multiplier for lighter model
)

# Freeze more layers for faster inference
for layer in base_model.layers[:-20]:  # Changed from -30
    layer.trainable = False

for layer in base_model.layers[-20:]:
    layer.trainable = True

# Simpler architecture
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)  # Reduced from 256
x = Dropout(0.3)(x)
output = Dense(train_data.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=4,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=0.00001
)

# Train
print("🚀 Starting training with OPTIMIZED model...")
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=test_data,
    callbacks=[early_stop, reduce_lr]
)

# Save as optimized model
model.save("fruit_model.keras", save_format='keras')

# Also save as H5 for better compatibility
model.save("fruit_model.h5")

print("✅ Model saved!")
print(f"📊 Training Accuracy: {history.history['accuracy'][-1] * 100:.2f}%")
print(f"📊 Testing Accuracy: {history.history['val_accuracy'][-1] * 100:.2f}%")
