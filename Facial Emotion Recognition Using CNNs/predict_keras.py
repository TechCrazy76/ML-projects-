import argparse
import cv2
import numpy as np
from os.path import isfile
from tensorflow.keras.models import load_model
from constants import *
from PIL import Image

def preprocess_face(gray_img, face):
    # crop -> resize -> scale -> reshape for model
    x, y, w, h = face
    face_img = gray_img[y:y+h, x:x+w]
    try:
        face_img = cv2.resize(face_img, (SIZE_FACE, SIZE_FACE), interpolation=cv2.INTER_CUBIC) / 255.0
    except Exception:
        return None
    return face_img.reshape(1, SIZE_FACE, SIZE_FACE, 1)

def pick_largest_face(faces):
    if len(faces) == 0:
        return None
    max_f = faces[0]
    for f in faces:
        if f[2]*f[3] > max_f[2]*max_f[3]:
            max_f = f
    return tuple(max_f)

def draw_prob_bars(img, probs, emotions, origin=(10,10)):
    # draw list of bars and labels on image (left side)
    x0, y0 = origin
    h_step = 20
    for i, (e, p) in enumerate(zip(emotions, probs)):
        y = y0 + i*h_step
        label = f"{e}: {p*100:.1f}%"
        cv2.putText(img, label, (x0, y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)
        cv2.rectangle(img, (x0+120, y+4), (x0+120+int(p*100), y+16), (255,0,0), -1)

def main(args):
    image_path = args.image
    model_path = args.model

    if not isfile(image_path):
        print("Image not found:", image_path); return
    if not isfile(model_path):
        print("Model not found:", model_path); return

    # load model
    model = load_model(model_path)

    # read image
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to read image:", image_path); return

    # grayscale for detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # face detection
    cascade = cv2.CascadeClassifier(CASC_PATH)
    faces = cascade.detectMultiScale(gray, scaleFactor=SCALEFACTOR, minNeighbors=5)
    face = pick_largest_face(faces)
    if face is None:
        print("No face detected.")
        # save original for reference
        cv2.imwrite('detected_faces_keras.png', image)
        return

    x,y,w,h = face
    # draw rectangle on original image
    cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)

    # preprocess and predict
    inp = preprocess_face(gray, face)
    if inp is None:
        print("Problem preprocessing face.")
        cv2.imwrite('detected_faces_keras.png', image)
        return

    probs = model.predict(inp)[0]
    top_idx = int(np.argmax(probs))
    top_emotion = EMOTIONS[top_idx]
    top_prob = probs[top_idx]

    # overlay predicted emotion near face
    text = f"{top_emotion} ({top_prob*100:.1f}%)"
    cv2.putText(image, text, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # draw probability bars on left
    draw_prob_bars(image, probs, EMOTIONS, origin=(10,10))

    # write result
    out_path = args.output
    cv2.imwrite(out_path, image)
    print("Saved:", out_path)
    # print probabilities
    for e, p in zip(EMOTIONS, probs):
        print(f"{e:10s}: {p*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict emotion from image using Keras model")
    parser.add_argument("--image", type=str, default="pics/1happy.jpg", help="path to input image")
    parser.add_argument("--model", type=str, default="data/Gudi_model_100_epochs_20000_faces_keras.h5", help="path to keras .h5 model")
    parser.add_argument("--output", type=str, default="detected_faces_keras.png", help="output image path")
    args = parser.parse_args()
    main(args)
