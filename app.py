from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import io
import cv2

from gradcam import generate_gradcam, overlay_heatmap, make_occlusion_heatmap

app = Flask(__name__)

IMG_SIZE = 128

# ✅ Load model
model = tf.keras.models.load_model("image_authenticity_model.keras")

# ✅ Force build (avoids Grad-CAM issues)
dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3))
model.predict(dummy)


# ✅ Preprocess image
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    image_data = None
    heatmap_data = None

    if request.method == "POST":
        file = request.files.get("image")

        if not file:
            return "No file uploaded"

        image = Image.open(file).convert("RGB")

        # ✅ Encode ORIGINAL image (no resize → better UI)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        image_data = base64.b64encode(buffer.getvalue()).decode()

        # ✅ Preprocess for model
        processed = preprocess_image(image)

        # ✅ Prediction
        prediction = float(model.predict(processed)[0][0])

        if prediction > 0.5:
            result = "AI Generated"
            confidence = round(prediction * 100, 2)
        else:
            result = "Real Image"
            confidence = round((1 - prediction) * 100, 2)

        # ✅ Grad-CAM
        try:
            heatmap = generate_gradcam(model, processed)
        except Exception as e:
            print("GradCAM Error:", e)
            heatmap = np.zeros((IMG_SIZE, IMG_SIZE))

        # ✅ Use resized ORIGINAL (better than processed image)
        original = np.array(image.resize((IMG_SIZE, IMG_SIZE)), dtype=np.uint8)

        # ✅ Overlay heatmap
        overlay = overlay_heatmap(original, heatmap)

        # ✅ Encode heatmap image properly
        success, buffer = cv2.imencode(".jpg", overlay)
        if success:
            heatmap_data = base64.b64encode(buffer.tobytes()).decode()
        else:
            heatmap_data = None

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image_data=image_data,
        heatmap_data=heatmap_data
    )


if __name__ == "__main__":
    app.run(debug=True)