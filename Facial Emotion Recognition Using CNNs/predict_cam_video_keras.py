import cv2
import numpy as np
from tensorflow.keras.models import load_model
from constants import *
from os.path import isfile
from PIL import Image, ImageFont, ImageDraw

# Load face cascade (uses your existing haarcascades folder)
face_cascade = cv2.CascadeClassifier(CASC_PATH)

def crop_and_preprocess(frame, face):
    x,y,w,h = face
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_img = gray[y:y+h, x:x+w]
    try:
        face_img = cv2.resize(face_img, (SIZE_FACE, SIZE_FACE), interpolation=cv2.INTER_CUBIC) / 255.0
    except Exception:
        return None
    return face_img.reshape(1, SIZE_FACE, SIZE_FACE, 1)

def draw_overlay(frame, probs, face):
    # Draw predicted bars and top text
    x,y,w,h = face
    # top text
    top_idx = int(np.argmax(probs))
    text = f"{EMOTIONS[top_idx]} ({probs[0][top_idx]*100:.0f}%)"
    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # small vertical bars + labels left
    left_x = 10
    y0 = 10
    for i, e in enumerate(EMOTIONS):
        label = f"{e}: {probs[0][i]*100:4.0f}%"
        yloc = y0 + 20*i
        cv2.putText(frame, label, (left_x, yloc+12), cv2.FONT_HERSHEY_PLAIN, 0.6, (0,255,0), 1)
        cv2.rectangle(frame, (left_x+140, yloc+4), (left_x+140+int(probs[0][i]*100), yloc+16), (255,0,0), -1)

def run(model_path, source=0, output_file=None, webcam=True):
    if not isfile(model_path):
        raise FileNotFoundError("Model not found: " + model_path)

    model = load_model(model_path)
    print("[+] Loaded model:", model_path)

    # source: 0 or video path
    if webcam:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(source)

    if output_file:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        out = cv2.VideoWriter(output_file, fourcc, 10, (width, height))
    else:
        out = None

    fps_time = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[+] End of video / no frame.")
            break

        # detect faces (use grayscale for cascade)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=SCALEFACTOR, minNeighbors=5)
        if len(faces) > 0:
            # pick largest
            largest = faces[0]
            for f in faces:
                if f[2]*f[3] > largest[2]*largest[3]:
                    largest = f
            x,y,w,h = largest
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

            inp = crop_and_preprocess(frame, largest)
            if inp is not None:
                probs = model.predict(inp)
                # overlay
                draw_overlay(frame, probs, largest)

        # draw FPS
        cur = cv2.getTickCount()
        # show frame
        cv2.imshow("Emotion (press q to quit)", frame)
        if out is not None:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    print("[+] Finished.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/Gudi_model_100_epochs_20000_faces_keras.h5")
    parser.add_argument("--source", default="0", help="0 for webcam or path to video file")
    parser.add_argument("--output", default="", help="optional output video path (e.g. out.avi)")
    args = parser.parse_args()

    webcam_flag = (args.source == "0")
    source_arg = 0 if webcam_flag else args.source
    out_arg = args.output if args.output.strip() else None

    try:
        run(args.model, source=source_arg, output_file=out_arg, webcam=webcam_flag)
    except Exception as e:
        print("Error:", e)
