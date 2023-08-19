import cv2
import numpy as np
import dlib
from imutils import face_utils
from flask import Flask, render_template, request
import threading

app = Flask(__name__)
cap = None
stop_thread = False
drowsiness_thread = None

@app.route('/')
def index():
    return render_template('index.html')

def detect_drowsiness():
    global cap, stop_thread

    # Your drowsiness detection logic...
    # Initializing the face detector and landmark detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

    # Drowsiness detection variables
    sleep = 0
    drowsy = 0
    active = 0
    status = ""
    color = (0, 0, 0)

    def compute(ptA, ptB):
        dist = np.linalg.norm(ptA - ptB)
        return dist

    def blinked(a, b, c, d, e, f):
        up = compute(b, d) + compute(c, e)
        down = compute(a, f)
        ratio = up / (2.0 * down)

        # Checking if it is blinked
        if (ratio > 0.25):
            return 2
        elif (ratio > 0.21 and ratio <= 0.25):
            return 1
        else:
            return 0

    while not stop_thread:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        face_frame = frame.copy()

        # Your face detection and drowsiness logic here
        # detected face in faces array
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            landmarks = predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)
            # The numbers are actually the landmarks which will show eye
            left_blink = blinked(landmarks[36], landmarks[37],landmarks[38], landmarks[41], landmarks[40], landmarks[39])
            right_blink = blinked(landmarks[42], landmarks[43],landmarks[44], landmarks[47], landmarks[46], landmarks[45])
            # Now judge what to do for the eye blinks
            if (left_blink == 0 or right_blink == 0):
                sleep += 1
                drowsy = 0
                active = 0
                if (sleep > 6):
                    status = "SLEEPING !!!"
                    color = (255, 0, 0)
            elif (left_blink == 1 or right_blink == 1):
                sleep = 0
                active = 0
                drowsy += 1
                if (drowsy > 6):
                    status = "Drowsy !"
                    color = (0, 0, 255)
            else:
                drowsy = 0
                sleep = 0
                active += 1
                if (active > 6):
                    status = "Active :)"
                    color = (0, 255, 0)
            cv2.putText(frame, status, (100, 100),cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            for n in range(0, 68):
                (x, y) = landmarks[n]
                cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)
        cv2.imshow("Frame", face_frame)
        cv2.imshow("Result of detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/Drowsiness_Extension')
def drowsiness():
    global cap, stop_thread, drowsiness_thread

    # Initialize the cap object every time the app is launched
    cap = cv2.VideoCapture(0)

    stop_thread = False
    drowsiness_thread = threading.Thread(target=detect_drowsiness)
    drowsiness_thread.start()

    return render_template('index.html')


@app.route('/terminate_App')
def terminate_app():
    global stop_thread, drowsiness_thread, cap

    stop_thread = True
    if drowsiness_thread is not None:
        drowsiness_thread.join()

    if cap is not None:
        cap.release()
        cap = None  # Reset the cap object

    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
# v?BiWqfvx62FNeb
