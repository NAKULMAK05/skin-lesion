import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend to avoid Tkinter usage

import os
import io
import base64
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_originimport matplotlib
matplotlib.use("Agg")  # Use non-interactive backend

import os
import io
import base64
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import tensorflow as tf
import joblib

# Load your models
segmentation_model = tf.keras.models.load_model("segmentation_model.h5")
classification_model = tf.keras.models.load_model("not_overfitting.h5")
nlp_model = joblib.load("path_to_trained_model.pkl")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://skin-lesion-classifier-frontend.vercel.app"}})

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CLASS_LABELS = ['bkl', 'nv', 'df', 'mel', 'vasc', 'bcc', 'akiec']

@app.after_request
def add_cors_headers(response):
    response.headers.add("Access-Control-Allow-Origin", "https://skin-lesion-classifier-frontend.vercel.app")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS")
    return response

@app.route("/predict", methods=["POST", "OPTIONS"])
@cross_origin(origins="https://skin-lesion-classifier-frontend.vercel.app")
def predict():
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        response.headers["Access-Control-Allow-Origin"] = "https://skin-lesion-classifier-frontend.vercel.app"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
        return response

    try:
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        image_file = request.files["image"]
        image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
        image_file.save(image_path)

        dx_type = request.form.get("dx_type", "")
        age = request.form.get("age", "")
        sex = request.form.get("sex", "")
        localization = request.form.get("localization", "")

        orig_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if orig_img is None:
            raise ValueError("Could not read image from path: " + image_path)
        orig_img = cv2.resize(orig_img, (224, 224))
        orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        img_norm = orig_img.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_norm, axis=0)

        mask_pred = segmentation_model.predict(img_batch)[0]
        mask_bin = (mask_pred > 0.5).astype(np.float32).squeeze()

        mask_3ch = np.repeat(mask_bin[:, :, np.newaxis], 3, axis=-1)
        masked_img = img_norm * mask_3ch

        cnn_probs = classification_model.predict(np.expand_dims(masked_img, axis=0))[0]
        top_idx = int(np.argmax(cnn_probs))
        top_class_label = CLASS_LABELS[top_idx] if top_idx < len(CLASS_LABELS) else "Unknown"
        top_class_conf = float(cnn_probs[top_idx])
        all_class_probs_str = ", ".join(
            f"{CLASS_LABELS[i]}: {cnn_probs[i]:.4f}" for i in range(len(CLASS_LABELS))
        )

        input_df = pd.DataFrame([{
            "dx_type": dx_type,
            "age": age,
            "sex": sex,
            "localization": localization
        }])
        nlp_pred = nlp_model.predict(input_df)[0]

        try:
            nlp_pred_numeric = float(nlp_pred)
            combined_pred = (cnn_probs + nlp_pred_numeric) / 2.0
            final_idx = int(np.argmax(combined_pred))
            final_label = CLASS_LABELS[final_idx]
        except (ValueError, TypeError):
            final_label = str(nlp_pred)

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs[0, 0].imshow(masked_img)
        axs[0, 0].axis("off")
        axs[0, 0].set_title("CNN", fontsize=12, pad=10)

        axs[0, 1].imshow(orig_img_rgb)
        axs[0, 1].axis("off")
        axs[0, 1].set_title("NLP", fontsize=12, pad=10)

        overlay = orig_img_rgb.copy()
        overlay[mask_bin > 0.5] = [255, 0, 0]
        axs[1, 0].imshow(overlay)
        axs[1, 0].axis("off")
        axs[1, 0].set_title("Segmentation", fontsize=12, pad=10)

        axs[1, 1].imshow(orig_img_rgb)
        axs[1, 1].axis("off")
        axs[1, 1].set_title("Combined", fontsize=12, pad=10)

        plt.tight_layout(pad=2.0)

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)
        plot_image_base64 = base64.b64encode(buf.getvalue()).decode()

        cnn_output = f"Predicted Class: {top_class_label} (Confidence: {top_class_conf:.2f})"
        nlp_output = f"NLP Prediction: {nlp_pred}"
        segmentation_output = "U-Net Segmentation Applied"
        final_output = f"This disease is most probably classified as: {final_label}"

        return jsonify({
            "plot_image": f"data:image/png;base64,{plot_image_base64}",
            "cnn_output": cnn_output,
            "all_class_probabilities": all_class_probs_str,
            "nlp_output": nlp_output,
            "segmentation_output": segmentation_output,
            "final_output": final_output
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Expose the Flask app as the serverless function handler for Vercel
handler = app

import tensorflow as tf
import joblib

# Load your models
segmentation_model = tf.keras.models.load_model("segmentation_model.h5")
classification_model = tf.keras.models.load_model("not_overfitting.h5")
nlp_model = joblib.load("path_to_trained_model.pkl")

app = Flask(__name__)
# Configure CORS to allow your frontend domain
CORS(app, resources={r"/*": {"origins": "https://skin-lesion-classifier-frontend.vercel.app"}})

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Example class labels for your CNN
CLASS_LABELS = ['bkl', 'nv', 'df', 'mel', 'vasc', 'bcc', 'akiec']

# Ensure that every response includes the CORS headers
@app.after_request
def add_cors_headers(response):
    response.headers.add("Access-Control-Allow-Origin", "https://skin-lesion-classifier-frontend.vercel.app")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS")
    return response

@app.route("/predict", methods=["POST", "OPTIONS"])
@cross_origin(origins="https://skin-lesion-classifier-frontend.vercel.app")
def predict():
    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        response.headers["Access-Control-Allow-Origin"] = "https://skin-lesion-classifier-frontend.vercel.app"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
        return response

    try:
        # 1) Check for image in the request
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        # 2) Save the uploaded image
        image_file = request.files["image"]
        image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
        image_file.save(image_path)

        # 3) Gather additional metadata
        dx_type = request.form.get("dx_type", "")
        age = request.form.get("age", "")
        sex = request.form.get("sex", "")
        localization = request.form.get("localization", "")

        # 4) Read and preprocess the image
        orig_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if orig_img is None:
            raise ValueError("Could not read image from path: " + image_path)
        orig_img = cv2.resize(orig_img, (224, 224))
        orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        img_norm = orig_img.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_norm, axis=0)  # Shape: (1, 224, 224, 3)

        # 5) Perform segmentation using U-Net model
        mask_pred = segmentation_model.predict(img_batch)[0]  # (224, 224, 1)
        mask_bin = (mask_pred > 0.5).astype(np.float32).squeeze()  # (224, 224)

        # 6) Create masked image for CNN classification
        mask_3ch = np.repeat(mask_bin[:, :, np.newaxis], 3, axis=-1)
        masked_img = img_norm * mask_3ch

        # 7) CNN classification
        cnn_probs = classification_model.predict(np.expand_dims(masked_img, axis=0))[0]
        top_idx = int(np.argmax(cnn_probs))
        top_class_label = CLASS_LABELS[top_idx] if top_idx < len(CLASS_LABELS) else "Unknown"
        top_class_conf = float(cnn_probs[top_idx])
        all_class_probs_str = ", ".join(
            f"{CLASS_LABELS[i]}: {cnn_probs[i]:.4f}" for i in range(len(CLASS_LABELS))
        )

        # 8) NLP prediction using metadata
        input_df = pd.DataFrame([{
            "dx_type": dx_type,
            "age": age,
            "sex": sex,
            "localization": localization
        }])
        nlp_pred = nlp_model.predict(input_df)[0]  # Could be numeric or a label

        # 9) Combine predictions (if applicable)
        try:
            nlp_pred_numeric = float(nlp_pred)
            combined_pred = (cnn_probs + nlp_pred_numeric) / 2.0
            final_idx = int(np.argmax(combined_pred))
            final_label = CLASS_LABELS[final_idx]
        except (ValueError, TypeError):
            final_label = str(nlp_pred)

        # 10) Build the figure for visualization (2x2 subplots)
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs[0, 0].imshow(masked_img)
        axs[0, 0].axis("off")
        axs[0, 0].set_title("CNN", fontsize=12, pad=10)

        axs[0, 1].imshow(orig_img_rgb)
        axs[0, 1].axis("off")
        axs[0, 1].set_title("NLP", fontsize=12, pad=10)

        overlay = orig_img_rgb.copy()
        overlay[mask_bin > 0.5] = [255, 0, 0]
        axs[1, 0].imshow(overlay)
        axs[1, 0].axis("off")
        axs[1, 0].set_title("Segmentation", fontsize=12, pad=10)

        axs[1, 1].imshow(orig_img_rgb)
        axs[1, 1].axis("off")
        axs[1, 1].set_title("Combined", fontsize=12, pad=10)

        plt.tight_layout(pad=2.0)

        # 11) Encode the figure as a base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)
        plot_image_base64 = base64.b64encode(buf.getvalue()).decode()

        # 12) Build textual outputs
        cnn_output = f"Predicted Class: {top_class_label} (Confidence: {top_class_conf:.2f})"
        nlp_output = f"NLP Prediction: {nlp_pred}"
        segmentation_output = "U-Net Segmentation Applied"
        final_output = f"This disease is most probably classified as: {final_label}"

        return jsonify({
            "plot_image": f"data:image/png;base64,{plot_image_base64}",
            "cnn_output": cnn_output,
            "all_class_probabilities": all_class_probs_str,
            "nlp_output": nlp_output,
            "segmentation_output": segmentation_output,
            "final_output": final_output
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Expose the Flask app as the handler for Vercel
handler = app
