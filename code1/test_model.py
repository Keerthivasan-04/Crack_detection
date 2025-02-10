import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load trained model
model = tf.keras.models.load_model("unet_crack_detector.h5")

# Function to preprocess an image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
    if img is None:
        raise ValueError(f"Error loading image: {image_path}")
    img_resized = cv2.resize(img, (256, 256))
    img_normalized = img_resized / 255.0  # Normalize
    img_normalized = np.expand_dims(img_normalized, axis=-1)  # Add channel dimension (H, W, 1)
    img_normalized = np.expand_dims(img_normalized, axis=0)  # Add batch dimension (1, H, W, 1)
    return img, img_normalized  # Return both original and preprocessed image

# Test the model with cracked and non-cracked images
image_paths = [
    r"C:\Users\akeer\OneDrive\Desktop\Final year Project\Code\dataset\cracked\cracked_sample.jpg",
    r"C:\Users\akeer\OneDrive\Desktop\Final year Project\Code\dataset\non_cracked\non_cracked_sample.jpg"
]

for image_path in image_paths:
    original_img, preprocessed_img = preprocess_image(image_path)
    prediction = model.predict(preprocessed_img)
    
    # Determine classification
    result = "Crack Detected" if prediction > 0.2 else "No Crack Detected"
    print(f"Prediction for {image_path}: {result}")
    
    # Display the image with prediction result
    plt.figure()
    plt.imshow(original_img, cmap='gray')
    plt.title(result)
    plt.axis('off')

plt.show()
