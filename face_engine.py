
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Add RetinaFace repository to Python path
ROOT = Path(__file__).resolve().parent
RETINAFACE_DIR = Path("C:/Users/LENOVO/Desktop/POC-Copy1/TALKFEVER/retinaface-pytorch")

sys.path.insert(0, str(RETINAFACE_DIR))

from config import get_config
from models import RetinaFace
from layers import PriorBox
from utils.box_utils import decode, decode_landmarks, nms


DEVICE = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )

WEIGHTS = Path("C:/Users/LENOVO/Desktop/POC-Copy1/TALKFEVER/retinaface-pytorch/weights/retinaface_r34.pth")
SFACE_MODEL = Path("C:/Users/LENOVO/Desktop/POC-Copy1/TALKFEVER/Sface/face_recognition_sface_2021dec.onnx")


def load_models():

    # Load RetinaFace ResNet34
    cfg = get_config("resnet34")

    detector = RetinaFace(cfg=cfg).to(DEVICE)

    state_dict = torch.load(
        str(WEIGHTS),
        map_location=DEVICE,
        weights_only=True
    )

    detector.load_state_dict(state_dict)
    detector.eval()

    # Load SFace
    recognizer = cv2.FaceRecognizerSF.create(
        str(SFACE_MODEL),
        ""
    )

    return detector, recognizer


def detect_face(image, detector):

    original = image.copy()

    height, width = image.shape[:2]

    # RetinaFace preprocessing
    img = image.astype(np.float32)
    img -= (104, 117, 123)

    img = img.transpose(2, 0, 1)
    tensor = torch.from_numpy(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        loc, conf, landmarks = detector(tensor)

    loc = loc.squeeze(0)
    conf = conf.squeeze(0)
    landmarks = landmarks.squeeze(0)

    cfg = get_config("resnet34")

    # Generate anchor boxes
    priorbox = PriorBox(
        cfg,
        image_size=(height, width)
    )

    priors = priorbox.generate_anchors().to(DEVICE)

    # Decode face boxes and landmarks
    boxes = decode(loc, priors, cfg["variance"])

    landmarks = decode_landmarks(
        landmarks,
        priors,
        cfg["variance"]
    )

    box_scale = torch.tensor(
        [width, height, width, height],
        device=DEVICE
    )

    landmark_scale = torch.tensor(
        [width, height] * 5,
        device=DEVICE
    )

    boxes = (boxes * box_scale).cpu().numpy()
    landmarks = (landmarks * landmark_scale).cpu().numpy()

    scores = conf.cpu().numpy()[:, 1]

    # Filter weak detections
    indexes = scores > 0.5

    boxes = boxes[indexes]
    landmarks = landmarks[indexes]
    scores = scores[indexes]

    if len(scores) == 0:
        raise ValueError("No face detected. Please try another photo.")

    # Keep strongest detection
    order = scores.argsort()[::-1][:5000]

    boxes = boxes[order]
    landmarks = landmarks[order]
    scores = scores[order]

    detections = np.hstack(
        (boxes, scores[:, None])
    ).astype(np.float32)

    keep = nms(detections, 0.4)

    detections = detections[keep]
    landmarks = landmarks[keep]

    # Select highest-confidence face
    best_box = detections[0]
    best_landmarks = landmarks[0]

    # SFace input format:
    # x, y, width, height, five landmark points
    x1, y1, x2, y2, confidence = best_box

    face = np.concatenate([
        [
            x1,
            y1,
            x2 - x1,
            y2 - y1
        ],
        best_landmarks
    ]).astype(np.float32)

    return original, face, confidence


def get_embedding(image, detector, recognizer):

    # Detect face
    original, face, confidence = detect_face(
        image,
        detector
    )

    # Align face using RetinaFace landmarks
    aligned_face = recognizer.alignCrop(
        original,
        face
    )

    # Generate SFace embedding
    embedding = recognizer.feature(aligned_face)

    embedding = embedding.flatten().astype(np.float32)

    return embedding, aligned_face, confidence


def cosine_similarity(embedding1, embedding2):

    embedding1 = embedding1 / (
        np.linalg.norm(embedding1) + 1e-10
    )

    embedding2 = embedding2 / (
        np.linalg.norm(embedding2) + 1e-10
    )

    score = np.dot(embedding1, embedding2)

    return float(score)



def verify_faces(image1, image2, detector, recognizer):

    embedding1, aligned1, confidence1 = get_embedding(
        image1, detector, recognizer
    )

    embedding2, aligned2, confidence2 = get_embedding(
        image2, detector, recognizer
    )

    score = cosine_similarity(embedding1, embedding2)

    if score >= 0.40:
        status = "Auto-Match"

    elif score >= 0.28:
        status = "Needs Review"

    else:
        status = "No Match"

    return {
        "score": score,
        "status": status,
        "selfie_embedding": embedding1,
        "profile_embedding": embedding2,
        "aligned_selfie": aligned1,
        "aligned_profile": aligned2,
        "selfie_confidence": float(confidence1),
        "profile_confidence": float(confidence2)
    }
