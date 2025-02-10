import cv2
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("unet_crack_detector.h5", compile=False)

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0  # Normalize
    return img[np.newaxis, ..., np.newaxis]

def edge_detection(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    edges = cv2.Canny(img, 50, 150)
    cv2.imwrite("edge_output.png", edges)
    return edges

def compare_results(image_path):
    pred_mask = predict_crack(image_path)
    edge_img = edge_detection(image_path)
    combined = cv2.addWeighted(pred_mask, 0.6, edge_img, 0.4, 0)
    cv2.imwrite("comparison.png", combined)

# Run test
compare_results("test_image.jpg")
