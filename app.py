from flask import Flask, request, render_template, Response
import os
import zipfile
from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import base64
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from deepface import DeepFace

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Set up upload and prediction folders
UPLOAD_FOLDER = './upload/'
PREDICTION_FOLDER = './predictions'
OUTPUT_FOLDER = './output'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICTION_FOLDER'] = PREDICTION_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Create the directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICTION_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Home page route
@app.route('/')
def home():
    return render_template('index.html')

# Extract ZIP file
def extract_zipfile(filepath, extractpath):
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(extractpath)

# Route to upload ZIP and extract contents
@app.route('/upload_zip', methods=['POST'])
def upload_zip():
    if 'zipfile' not in request.files:
        return 'No file part'
    
    file = request.files['zipfile']
    if file.filename == '':
        return 'No selected file'
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        extractpath = os.path.join(app.config['UPLOAD_FOLDER'], 'extracted')
        file.save(filepath)
        extract_zipfile(filepath, extractpath)
        return render_template('index.html', uploaded="Dataset uploaded successfully")

# Function to generate face embeddings
def generate_embeddings():
    print("Loading Face Detector...")
    protoPath = "face_detection_model/deploy.prototxt"
    modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    print("Loading Face Recognizer...")
    embedder = cv2.dnn.readNetFromTorch("face_detection_model/nn4.small2.v1.t7")

    imagePaths = list(paths.list_images("upload/extracted/dataset"))
    knownEmbeddings = []
    knownNames = []
    total = 0

    for (i, imagePath) in enumerate(imagePaths):
        if (i % 50 == 0):
            print(f"Processing image {i}/{len(imagePaths)}")
        name = imagePath.split(os.path.sep)[-2]
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]

        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        detector.setInput(imageBlob)
        detections = detector.forward()

        if len(detections) > 0:
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                if fW < 20 or fH < 20:
                    continue

                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())
                total += 1

    print(f"Serializing {total} encodings...")
    data = {"embeddings": knownEmbeddings, "names": knownNames}
    with open(os.path.join(app.config['OUTPUT_FOLDER'], "embeddings.pickle"), "wb") as f:
        pickle.dump(data, f)

# Route to train model
@app.route('/train_model', methods=['POST'])
def train_model():
    generate_embeddings()

    print("Loading face embeddings...")
    data = pickle.loads(open(os.path.join(app.config['OUTPUT_FOLDER'], "embeddings.pickle"), "rb").read())

    print("Encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])

    print("Training model...")
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    with open(os.path.join(app.config['OUTPUT_FOLDER'], "recognizer.pickle"), "wb") as f:
        pickle.dump(recognizer, f)

    with open(os.path.join(app.config['OUTPUT_FOLDER'], "le.pickle"), "wb") as f:
        pickle.dump(le, f)

    return render_template('index.html', trained="Model has been trained successfully")

# Function to make prediction on image
def make_prediction(image_path):
    print("Loading Face Detector...")
    protoPath = os.path.join('face_detection_model', "deploy.prototxt")
    modelPath = os.path.join('face_detection_model', "res10_300x300_ssd_iter_140000.caffemodel")
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    print("Loading Face Recognizer...")
    embedder = cv2.dnn.readNetFromTorch('face_detection_model/nn4.small2.v1.t7')

    recognizer = pickle.loads(open(os.path.join(app.config['OUTPUT_FOLDER'], "recognizer.pickle"), "rb").read())
    le = pickle.loads(open(os.path.join(app.config['OUTPUT_FOLDER'], "le.pickle"), "rb").read())

    image = cv2.imread(image_path)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),
                                      (104.0, 177.0, 123.0), swapRB=False, crop=False)
    detector.setInput(imageBlob)
    detections = detector.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            if fW < 20 or fH < 20:
                continue

            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            # Emotion detection using DeepFace
            emotion_result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
            dominant_emotion = emotion_result[0]['dominant_emotion']

            # Add both name and emotion to the output
            text = f"{name}: {proba * 100:.2f}% - {dominant_emotion}"
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    _, buffer = cv2.imencode('.jpg', image)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return img_str

# Route to upload an image for prediction
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return 'No image uploaded', 400
    
    image = request.files['image']
    if image.filename == '':
        return 'No image selected', 400
    
    filepath = os.path.join(app.config['PREDICTION_FOLDER'], image.filename)
    image.save(filepath)

    prediction_image = make_prediction(filepath)
    return render_template('index.html', prediction_image=prediction_image)

# Video feed with emotion detection
def emotion_detection_feed():
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Use DeepFace to analyze emotions in the frame
        emotion_result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        dominant_emotion = emotion_result[0]['dominant_emotion']

        # Display the dominant emotion on the video feed
        cv2.putText(frame, f"Emotion: {dominant_emotion}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route to stream the video feed
@app.route('/video_feed')
def video_feed():
    return Response(emotion_detection_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
