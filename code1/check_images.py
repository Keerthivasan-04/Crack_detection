import cv2  # <-- Add this line
import matplotlib.pyplot as plt
import os

# Define paths
cracked_image_path = r"C:\Users\akeer\OneDrive\Desktop\Final year Project\Code\dataset\cracked\cracked_sample.jpg"
non_cracked_image_path = r"C:\Users\akeer\OneDrive\Desktop\Final year Project\Code\dataset\non_cracked\non_cracked_sample.jpg"

# Load and display cracked image
cracked_img = cv2.imread(cracked_image_path)
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(cracked_img, cv2.COLOR_BGR2RGB))
plt.title("Cracked Image")

# Load and display non-cracked image
non_cracked_img = cv2.imread(non_cracked_image_path)
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(non_cracked_img, cv2.COLOR_BGR2RGB))
plt.title("Non-Cracked Image")

plt.show()
