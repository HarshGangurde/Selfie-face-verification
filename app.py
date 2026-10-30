
import streamlit as st
import cv2
import numpy as np
import os
import json

from PIL import Image
from io import BytesIO
from datetime import datetime

from face_engine import load_models, verify_faces


# Page configuration
st.set_page_config(
    page_title="Face Verification",
    page_icon="🔐",
    layout="wide"
)

st.title("🔐 Face Verification System")

st.write(
    "Take a selfie and upload your existing profile photo "
    "to verify whether both images belong to the same person."
)


# Load AI models only once
@st.cache_resource
def initialize_models():
    return load_models()


try:
    with st.spinner("Loading RetinaFace and SFace models..."):
        detector, recognizer = initialize_models()

    st.success("AI models loaded successfully!")

except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# Two input columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Take your selfie")

    selfie_file = st.camera_input(
        "Open camera and capture your face"
    )

with col2:
    st.subheader("2. Upload profile photo")

    profile_file = st.file_uploader(
        "Choose your existing profile photo",
        type=["jpg", "jpeg", "png"]
    )


# Convert uploaded images to OpenCV format
def convert_image(uploaded_file):

    image = Image.open(
        BytesIO(uploaded_file.getvalue())
    ).convert("RGB")

    image = np.array(image)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    return image


# Verification
if st.button(
    "Verify My Face",
    type="primary",
    use_container_width=True
):

    if selfie_file is None:
        st.warning("Please capture your selfie.")

    elif profile_file is None:
        st.warning("Please upload your profile photo.")

    else:

        try:

            selfie = convert_image(selfie_file)
            profile = convert_image(profile_file)

            with st.spinner(
                "Detecting faces and comparing embeddings..."
            ):

                result = verify_faces(
                    selfie,
                    profile,
                    detector,
                    recognizer
                )

            # Create unique folder for each verification
            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S_%f"
            )

            save_dir = os.path.join(
                "results",
                f"verification_{timestamp}"
            )

            os.makedirs(save_dir, exist_ok=True)

            # Save original images
            cv2.imwrite(
                os.path.join(save_dir, "original_selfie.jpg"),
                selfie
            )

            cv2.imwrite(
                os.path.join(save_dir, "original_profile.jpg"),
                profile
            )

            # Save aligned images used by SFace
            cv2.imwrite(
                os.path.join(save_dir, "aligned_selfie.jpg"),
                result["aligned_selfie"]
            )

            cv2.imwrite(
                os.path.join(save_dir, "aligned_profile.jpg"),
                result["aligned_profile"]
            )

            # Save embeddings
            np.save(
                os.path.join(save_dir, "selfie_embedding.npy"),
                result["selfie_embedding"]
            )

            np.save(
                os.path.join(save_dir, "profile_embedding.npy"),
                result["profile_embedding"]
            )

            # Save scores and verification result
            verification_data = {
                "timestamp": timestamp,
                "cosine_similarity": result["score"],
                "status": result["status"],
                "selfie_detection_confidence": result["selfie_confidence"],
                "profile_detection_confidence": result["profile_confidence"],
                "selfie_embedding": result["selfie_embedding"].tolist(),
                "profile_embedding": result["profile_embedding"].tolist()
            }

            with open(
                os.path.join(save_dir, "verification_result.json"),
                "w"
            ) as file:

                json.dump(
                    verification_data,
                    file,
                    indent=4
                )

            st.divider()

            st.subheader("Verification Results")

            image1, image2 = st.columns(2)

            with image1:

                st.write("Aligned Selfie")

                st.image(
                    cv2.cvtColor(
                        result["aligned_selfie"],
                        cv2.COLOR_BGR2RGB
                    )
                )

            with image2:

                st.write("Aligned Profile Photo")

                st.image(
                    cv2.cvtColor(
                        result["aligned_profile"],
                        cv2.COLOR_BGR2RGB
                    )
                )

            score = result["score"]
            status = result["status"]

            st.metric(
                "Cosine Similarity Score",
                f"{score:.4f}"
            )

            st.progress(
                max(0.0, min(1.0, score))
            )

            if status == "Auto-Match":

                st.success(
                    "✅ AUTO-MATCH: Face verified successfully!"
                )

            elif status == "Needs Review":

                st.warning(
                    "⚠️ NEEDS REVIEW: Please retry with better "
                    "lighting or send this case to admin review."
                )

            else:

                st.error(
                    "❌ NO MATCH: The faces did not meet "
                    "the required similarity threshold."
                )

            st.write("### Model Details")

            st.write(
                "Selfie detection confidence:",
                round(result["selfie_confidence"], 4)
            )

            st.write(
                "Profile detection confidence:",
                round(result["profile_confidence"], 4)
            )

        except Exception as e:

            st.error(str(e))
