import tensorflow as tf
import numpy as np
import pandas as pd

IMG_SIZE = (456, 456)
MODEL_PATH = '/Users/mac/Development/diploma/skin-mutation-detection/efficientnet-b5.keras'
LABELS = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

LABELS_CSV_PATH = '/Users/mac/Development/diploma/skin-mutation-detection/ham10000/ham10000_Test_GroundTruth_Updated.csv'  # Update this with the actual path
labels_df = pd.read_csv(LABELS_CSV_PATH)

def get_actual_label(image_id, labels_df):
    row = labels_df[labels_df['image'] == image_id].iloc[0]
    actual_label = row[1:].idxmax()
    return actual_label

print("-" * 150)
print("\n")
print("Loading model...\n\n")
print("-" * 150)
model = tf.keras.models.load_model(MODEL_PATH)

def load_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE[0], IMG_SIZE[1]])
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img

print("-" * 150)
print("\n")
print("\n")
path = input("Please enter a path to the image: ")
print("\n")
print("\n")

img = load_image(path)
img = tf.expand_dims(img, axis=0)

print("-" * 150)
print("\n\n")
print("Predicting mutation class...\n\n")
print("\n\n")

prediction = model.predict(img, batch_size=1)

print("\n\n")
print("-" * 150)
print("\n\n")
predicted_label = LABELS[np.argmax(prediction, axis=1)[0]]
print(f"Image Path: {path}")
print(f"Predicted Label: {predicted_label}")

image_id = path.split('/')[-1].split('.')[0]
actual_label = get_actual_label(image_id, labels_df)

print(f"Actual Label: {actual_label}")
print("\n\n")
print("-" * 150)
