import tensorflow as tf
import numpy as np
import cv2
import sys

# Load trained model
model = tf.keras.models.load_model("crack_detection_model.h5")

def preprocess_image(image_path, size=(256, 256)):
    """Preprocess the image: resize, normalize, and reshape."""
    img = cv2.imread(image_path)
    img = cv2.resize(img, size)
    img = img / 255.0  # Normalize
    return img[np.newaxis, ...]

def predict_crack(image_path):
    """Predict whether the given image has a crack or not."""
    img = preprocess_image(image_path)
    pred_mask = model.predict(img)[0]  # Get predicted mask
    
    # Convert to binary mask (threshold 0.5)
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255

    # Save the predicted mask
    cv2.imwrite("predicted_mask.png", pred_mask)
    
    # Calculate the percentage of cracked area
    crack_percentage = (np.sum(pred_mask) / (256 * 256 * 255)) * 100
    
    # Define threshold for classification
    threshold = 5  # If more than 5% of the image is cracked, classify as cracked
    classification = "Cracked" if crack_percentage > threshold else "Non-Cracked"

    print(f"Classification: {classification}")
    print(f"Crack Area Percentage: {crack_percentage:.2f}%")

    return classification

# Test the function with an image
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python classify_crack.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    classification = predict_crack(image_path)
    print(f"The given image is classified as: {classification}")
