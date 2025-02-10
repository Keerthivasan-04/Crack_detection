import matplotlib.pyplot as plt
import pickle

# Load training history
with open("training_history.pkl", "rb") as f:
    history = pickle.load(f)

plt.figure(figsize=(8, 5))
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()
