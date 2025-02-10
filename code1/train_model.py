import cv2
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load images function
def load_images(folder, cracked=True):
    images, masks = [], []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (256, 256))
            images.append(img / 255.0)  # Normalize
            masks.append(1 if cracked else 0)  # Label: 1 for cracked, 0 for non-cracked
    return np.array(images), np.array(masks)

# Load dataset
X_cracked, Y_cracked = load_images(r"C:\Users\akeer\OneDrive\Desktop\Final year Project\Code\dataset\cracked", cracked=True)
X_non_cracked, Y_non_cracked = load_images(r"C:\Users\akeer\OneDrive\Desktop\Final year Project\Code\dataset\non_cracked", cracked=False)

# Combine data
X = np.concatenate((X_cracked, X_non_cracked), axis=0)
Y = np.concatenate((Y_cracked, Y_non_cracked), axis=0)

# Reshape input for CNN
X = X.reshape(-1, 256, 256, 1)

# Split into training and validation
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification (crack or no crack)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model and track accuracy/loss
history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=10, batch_size=4)

# Save the trained model
model.save(r"C:\Users\akeer\OneDrive\Desktop\Final year Project\Code\code1\unet_crack_detector.h5")
print("Model trained and saved successfully.")

# Plot training performance
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
