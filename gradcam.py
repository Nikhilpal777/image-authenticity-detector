import tensorflow as tf
import numpy as np
import cv2


# ✅ Get last Conv layer automatically
def get_last_conv_layer(model):
    for layer in reversed(model.layers):

        if isinstance(layer, tf.keras.Sequential):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return sub_layer

        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer

    raise ValueError("No Conv2D layer found")


# ✅ Grad-CAM (Smooth Heatmap)
def generate_gradcam(model, img_array):

    img_array = tf.cast(img_array, tf.float32)

    last_conv_layer = get_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        tape.watch(img_array)

        conv_outputs, predictions = grad_model(img_array)

        # Binary or multi-class support
        if predictions.shape[-1] == 1:
            loss = predictions[:, 0]
        else:
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        return np.zeros((128, 128))

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    # Weighted sum of feature maps
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = heatmap.numpy()

    # ReLU (remove negative values)
    heatmap = np.maximum(heatmap, 0)

    # Normalize (VERY IMPORTANT)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)
    heatmap = np.clip(heatmap, 0.1, 1)

    # 🔥 Improve visibility
    heatmap = np.power(heatmap, 2.0)

    return heatmap


# ✅ Overlay heatmap on image
def overlay_heatmap(original_img, heatmap):

    # Resize heatmap to match image
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

    heatmap = np.uint8(255 * heatmap)

    # 🔥 Multicolor (blue → green → red)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Convert BGR → RGB
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = cv2.addWeighted(original_img, 0.4, heatmap, 0.6, 0)

    return overlay
def make_occlusion_heatmap(img_array, model, patch=32):

    img_size = img_array.shape[1]
    heatmap = np.zeros((img_size, img_size))

    base_pred = model.predict(img_array)[0][0]

    for y in range(0, img_size, patch):
        for x in range(0, img_size, patch):

            temp = img_array.copy()
            temp[:, y:y+patch, x:x+patch, :] = 0

            pred = model.predict(temp)[0][0]

            diff = abs(base_pred - pred)

            heatmap[y:y+patch, x:x+patch] = diff

    # Normalize
    heatmap -= np.min(heatmap)
    heatmap /= (np.max(heatmap) + 1e-8)

    return heatmap