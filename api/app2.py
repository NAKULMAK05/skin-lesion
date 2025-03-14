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
from flask_cors import CORS
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

# Example class labels for your CNN
# Make sure these are in the correct order as used during training.
CLASS_LABELS = ['bkl', 'nv', 'df', 'mel', 'vasc', 'bcc', 'akiec']

@app.route("/predict1", methods=["POST"])
def predict():
    try:
        # 1) Check for image
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        # 2) Save uploaded image
        image_file = request.files["image"]
        image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
        image_file.save(image_path)

        # 3) Gather metadata
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
        img_batch = np.expand_dims(img_norm, axis=0)  # shape: (1, 224, 224, 3)

        # 5) Segmentation
        mask_pred = segmentation_model.predict(img_batch)[0]  # (224, 224, 1)
        mask_bin = (mask_pred > 0.5).astype(np.float32).squeeze()  # (224, 224)

        # 6) Create masked image for CNN
        mask_3ch = np.repeat(mask_bin[:, :, np.newaxis], 3, axis=-1)
        masked_img = img_norm * mask_3ch

        # 7) CNN classification using reference snippet
        prediction = classification_model.predict(np.expand_dims(masked_img, axis=0))
        predicted_class_idx = np.argmax(prediction)
        predicted_class = CLASS_LABELS[predicted_class_idx]
        confidence_level = prediction[0][predicted_class_idx]

        # Build a string listing all class probabilities
        all_class_probs_str = ", ".join(
            f"{CLASS_LABELS[i]}: {prediction[0][i]:.4f}" for i in range(len(CLASS_LABELS))
        )

        # 8) NLP prediction
        input_df = pd.DataFrame([{
            "dx_type": dx_type,
            "age": age,
            "sex": sex,
            "localization": localization
        }])
        nlp_pred = nlp_model.predict(input_df)[0]  # could be numeric or string label

        # 9) Combine predictions
        # If NLP is numeric, combine with CNN. Otherwise, pick NLP as final.
        try:
            nlp_pred_numeric = float(nlp_pred)
            # If we get here, NLP is numeric, so combine it with CNN probabilities
            combined_pred = (prediction[0] + nlp_pred_numeric) / 2.0
            final_idx = int(np.argmax(combined_pred))
            final_label = CLASS_LABELS[final_idx]
        except (ValueError, TypeError):
            final_label = str(nlp_pred)

        # 10) Build the figure (2x2 subplots) with minimal text
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # Top-left: CNN masked image
        axs[0, 0].imshow(masked_img)
        axs[0, 0].axis("off")
        axs[0, 0].set_title("CNN", fontsize=12, pad=10)

        # Top-right: Original image (for NLP reference)
        axs[0, 1].imshow(orig_img_rgb)
        axs[0, 1].axis("off")
        axs[0, 1].set_title("NLP", fontsize=12, pad=10)

        # Bottom-left: U-Net segmentation overlay
        overlay = orig_img_rgb.copy()
        overlay[mask_bin > 0.5] = [255, 0, 0]
        axs[1, 0].imshow(overlay)
        axs[1, 0].axis("off")
        axs[1, 0].set_title("Segmentation", fontsize=12, pad=10)

        # Bottom-right: Combined prediction view
        axs[1, 1].imshow(orig_img_rgb)
        axs[1, 1].axis("off")
        axs[1, 1].set_title("Combined", fontsize=12, pad=10)

        plt.tight_layout(pad=2.0)

        # 11) Encode figure to base64
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)
        plot_image_base64 = base64.b64encode(buf.getvalue()).decode()

        # 12) Build textual outputs
        cnn_output = f"Predicted Class: {predicted_class} (Confidence: {confidence_level:.2f})"
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

if __name__ == "__main__":
    app.run(debug=True, port=5000)
