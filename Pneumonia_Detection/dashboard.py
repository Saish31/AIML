import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="Pneumonia Detection from X-Rays",
    page_icon="🫁",
    layout="wide"
)


# --- Caching the Model ---
@st.cache_resource
def load_model():
    """Loads the pre-trained model from file."""
    try:
        # Load the model without compiling it, as we only need it for inference
        model = tf.keras.models.load_model("pneumonia_classifier_model.keras", compile=False)
        return model
    except Exception as e:
        st.error(
            f"Error loading model: {e}. Please make sure the 'pneumonia_classifier_model.keras' file is in the same directory.")
        return None


model = load_model()


# --- Grad-CAM Functions ---
def get_img_array(img, size):
    """Prepares a PIL image for model prediction."""
    img = img.resize(size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return tf.keras.applications.vgg16.preprocess_input(array)


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, 0]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def display_gradcam(img, heatmap, alpha=0.6):
    """Overlays the heatmap on the original image."""
    img_np = np.array(img)
    heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img_np
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img


# --- Streamlit UI ---
st.title("🫁 Pneumonia Detection from Chest X-Rays")
st.write(
    "Upload a chest X-ray image. The model will predict whether it shows signs of pneumonia and generate a heatmap to explain its decision.")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpeg", "jpg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded X-Ray", use_column_width=True)

    with col2:
        with st.spinner("Analyzing image..."):
            IMG_SIZE = (224, 224)
            img_array = get_img_array(image, size=IMG_SIZE)

            # Make prediction
            prediction = model.predict(img_array)[0][0]
            confidence = prediction if prediction > 0.5 else 1 - prediction
            result = "PNEUMONIA" if prediction > 0.5 else "NORMAL"

            if result == "PNEUMONIA":
                st.error(f"**Result: {result}** (Confidence: {confidence:.2%})")
            else:
                st.success(f"**Result: {result}** (Confidence: {confidence:.2%})")

            # Generate and display Grad-CAM
            st.write("Generating Explainability Heatmap (Grad-CAM)...")

            # Create a temporary cloned model to remove the final activation for a better heatmap
            cloned_model = tf.keras.models.clone_model(model)
            cloned_model.set_weights(model.get_weights())
            cloned_model.layers[-1].activation = None
            last_conv_layer_name = "block5_conv3"  # The name of the base model layer

            heatmap = make_gradcam_heatmap(img_array, cloned_model, last_conv_layer_name)
            superimposed_img = display_gradcam(image, heatmap)
            st.image(superimposed_img, caption="Grad-CAM Explainability Overlay", use_column_width=True)

    st.info("The highlighted (hot) regions are what the model focused on most to make its prediction.", icon="💡")