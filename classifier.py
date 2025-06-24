import time
import os
import zipfile
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetV2S, DenseNet121, InceptionResNetV2, ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
import argparse
from keras.layers import TFSMLayer, Input

IMG_SIZE = (456, 456)
BATCH_SIZE = 32
EPOCHS = 20
LABELS = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

class SkinCancerClassifier:
    def __init__(self, model, csv_path):
        self.model = model
        self.labels_df = pd.read_csv(csv_path)
        self.labels_df['image'] = self.labels_df['image'].astype(str)

    def load_image(self, file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img

    def get_true_label(self, file_path):
        image_id = os.path.splitext(os.path.basename(file_path))[0]
        row = self.labels_df[self.labels_df['image'] == image_id]
        if row.empty:
            return "UNK"
        one_hot = row[LABELS].values[0]
        index = int(tf.argmax(one_hot).numpy())
        return LABELS[index]

    def predict(self, file_path):
        img = self.load_image(file_path)
        img = tf.expand_dims(img, axis=0)

        start_time = time.time()
        print(self.model)
        predictions = self.model.predict(img)

        predictions = list(predictions.values())[0][0]
        elapsed_time = time.time() - start_time
        predicted_index = tf.argmax(predictions).numpy()
        confidence = predictions[predicted_index]
        predicted_label = LABELS[predicted_index]
        true_label = self.get_true_label(file_path)

        return {
            "predicted_label": predicted_label,
            "confidence": confidence,
            "true_label": true_label,
            "inference_time_sec": elapsed_time
        }

class SkinCancerVisualizer:
    def __init__(self, classifier, img_size=(224, 224), save_path="/Users/mac/Development/diploma/skin-mutation-detection/runs/classify/test"):
        self.classifier = classifier
        self.img_size = img_size
        self.save_path = save_path

    def visualize(self, file_path):
        # Get prediction data
        result = self.classifier.predict(file_path)
        predicted_label = result["predicted_label"]
        true_label = result["true_label"]
        confidence = result["confidence"]
        time_taken = result["inference_time_sec"]

        raw_img = tf.io.read_file(file_path)
        raw_img = tf.image.decode_jpeg(raw_img, channels=3)
        raw_img = tf.image.resize(raw_img, self.img_size).numpy().astype("uint8")
        
        correct = predicted_label == true_label
        title_color = "green" if correct else "red"
        title_text = f"Sample Name: {os.path.basename(file_path)}\nPredicted: {predicted_label} ({confidence:.2%})\nTrue: {true_label}\nTime: {time_taken:.3f} sec"

        fig = plt.figure(figsize=(6, 6))
        fig.canvas.manager.set_window_title(f"{os.path.basename(file_path)}")
        plt.imshow(raw_img)
        plt.title(title_text, color=title_color, fontsize=12)
        plt.axis("off")
        plt.savefig(os.path.join(self.save_path, os.path.basename(file_path)), bbox_inches=None, pad_inches=0)
        plt.show()

def model(model_path):
    layer = layers.TFSMLayer(
        model_path,
        call_endpoint="serving_default"
    )
    input_tensor = Input(shape=(*IMG_SIZE, 3), name="input_image")
    output_tensor = layer(input_tensor)
    model = tf.keras.models.Model(inputs=input_tensor, outputs=output_tensor)

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Skin Cancer Classifier Visualization CLI")
    parser.add_argument('--model_path', type=str, default="/Users/mac/Development/diploma/skin-mutation-detection/models/weights/efficientnetv2-s-onehot-deep-unified-softmask-se-mish-finetune-05", help='Path to the trained Keras model (.h5)')
    parser.add_argument('--csv_path', type=str, help='Path to the ground truth CSV file', default="/Users/mac/Development/diploma/skin-mutation-detection/unified_dataset/gt.csv")
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image to classify and visualize')
    parser.add_argument('--save_dir', type=str, default="/Users/mac/Development/diploma/skin-mutation-detection/runs/classify/test", help='Directory where results will be saved')

    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")

    model = tf.keras.models.load_model("./model.keras")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    with open('model2.tflite', 'wb') as f:
        f.write(tflite_model)

    classifier = SkinCancerClassifier(model=model, csv_path=args.csv_path)
    visualizer = SkinCancerVisualizer(classifier=classifier, save_path=args.save_dir)
    classifier.predict(args.image_path) # warm up
    visualizer.visualize(args.image_path)