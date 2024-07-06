import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.utils import Sequence, to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart 
import datetime
from datetime import timezone


app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
MODEL_PATH = 'idr.h5'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

model = load_model(MODEL_PATH)

# Define paths and parameters
base_path = 'E:/pro_v/flask'
frame_size = (128, 128)  # Example frame size, adjust as needed
frame_count = 16  # Example frame count per video, adjust as needed
batch_size = 2
epochs = 20
learning_rate = 0.0001



class VideoDataGenerator(Sequence):
    def __init__(self, video_paths, labels, frame_size, frame_count, batch_size, augment=False):
        self.video_paths = video_paths
        self.labels = labels
        self.frame_size = frame_size
        self.frame_count = frame_count
        self.batch_size = batch_size
        self.augment = augment
        self.indices = np.arange(len(self.video_paths))
        self.augmentor = ImageDataGenerator(
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.2
        )

    def __len__(self):
        return int(np.ceil(len(self.video_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_video_paths = [self.video_paths[i] for i in batch_indices]
        batch_labels = [self.labels[i] for i in batch_indices]

        X, y = self.__data_generation(batch_video_paths, batch_labels)
        return X, y

    def __data_generation(self, batch_video_paths, batch_labels):
        X = np.empty((len(batch_video_paths), self.frame_count, *self.frame_size, 3), dtype=np.float32)
        y = np.empty((len(batch_video_paths)), dtype=int)

        for i, video_path in enumerate(batch_video_paths):
            frames = self.extract_frames(video_path)
            if self.augment:
                frames = self.augment_frames(frames)
            X[i,] = frames
            y[i] = batch_labels[i]

        return X, to_categorical(y, num_classes=2)

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = max(1, total_frames // self.frame_count)
        current_frame = 0

        while len(frames) < self.frame_count:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                break
            frame_resized = cv2.resize(frame, self.frame_size)
            frame_array = np.array(frame_resized) / 255.0
            frames.append(frame_array)
            current_frame += frame_step

        cap.release()

        while len(frames) < self.frame_count:
            frames.append(frames[-1])
            
        return np.array(frames)
    
    def augment_frames(self, frames):
        augmented_frames = []
        for frame in frames:
            frame = self.augmentor.random_transform(frame)
            augmented_frames.append(frame)
        return np.array(augmented_frames)

def load_video_paths_and_labels(base_path):
    subsets = ['train', 'validate', 'test']
    video_paths = {subset: [] for subset in subsets}
    labels = {subset: [] for subset in subsets}
    
    for subset in subsets:
        good_subset_path = os.path.join(base_path, 'normal', subset)
        leak_subset_path = os.path.join(base_path, 'leak', subset)
        
        if not os.path.exists(good_subset_path):
            raise FileNotFoundError(f"Path does not exist: {good_subset_path}")
        
        if not os.path.exists(leak_subset_path):
            raise FileNotFoundError(f"Path does not exist: {leak_subset_path}")
        
        for filename in os.listdir(good_subset_path):
            if filename.endswith(('.mp4', '.avi', '.mov', '.mkv','.webm')):
                video_paths[subset].append(os.path.join(good_subset_path, filename))
                labels[subset].append(0)
        
        for filename in os.listdir(leak_subset_path):
            if filename.endswith(('.mp4', '.avi', '.mov', '.mkv','.webm')):
                video_paths[subset].append(os.path.join(leak_subset_path, filename))
                labels[subset].append(1)

    return video_paths, labels

# Load video paths and labels
video_paths, labels = load_video_paths_and_labels(base_path)

# Create data generators
train_gen = VideoDataGenerator(video_paths['train'], labels['train'], frame_size, frame_count, batch_size, augment=True)
val_gen = VideoDataGenerator(video_paths['validate'], labels['validate'], frame_size, frame_count, batch_size, augment=False)
test_gen = VideoDataGenerator(video_paths['test'], labels['test'], frame_size, frame_count, batch_size, augment=False)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_gen)
# To calculate additional evaluation metrics
y_true = []
y_pred = []

for i in range(len(test_gen)):
    X, y = test_gen[i]
    predictions = model.predict(X)
    y_true.extend(np.argmax(y, axis=1))
    y_pred.extend(np.argmax(predictions, axis=1))

print('Confusion Matrix : ')
print(confusion_matrix(y_true, y_pred))
cm=confusion_matrix(y_true, y_pred)
# Extract and print each term of the confusion matrix
tn, fp, fn, tp = cm.ravel()

print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP): {tp}")
print('Classification Report Metrics : ')
print(classification_report(y_true, y_pred, target_names=['Good', 'Leaking'], zero_division=0))
# Generate the classification report as a dictionary
report_dict = classification_report(y_true, y_pred, target_names=['Good', 'Leaking'], output_dict=True, zero_division=0)
# Extract and print individual components
precision_good = report_dict['Good']['precision']
recall_good = report_dict['Good']['recall']
f1_good = report_dict['Good']['f1-score']
support_good = report_dict['Good']['support']
precision_leaking = report_dict['Leaking']['precision']
recall_leaking = report_dict['Leaking']['recall']
f1_leaking = report_dict['Leaking']['f1-score']
support_leaking = report_dict['Leaking']['support']
accuracy = report_dict['accuracy']
macro_avg_precision = report_dict['macro avg']['precision']
macro_avg_recall = report_dict['macro avg']['recall']
macro_avg_f1 = report_dict['macro avg']['f1-score']
weighted_avg_precision = report_dict['weighted avg']['precision']
weighted_avg_recall = report_dict['weighted avg']['recall']
weighted_avg_f1 = report_dict['weighted avg']['f1-score']
print(f"\nMetrics for 'Good':")
print(f"Precision: {precision_good}")
print(f"Recall: {recall_good}")
print(f"F1-Score: {f1_good}")
print(f"Support: {support_good}")
print(f"\nMetrics for 'Leaking':")
print(f"Precision: {precision_leaking}")
print(f"Recall: {recall_leaking}")
print(f"F1-Score: {f1_leaking}")
print(f"Support: {support_leaking}")
print(f"\nOverall Metrics:")
print(f"Accuracy: {accuracy}")
print(f"Macro Avg Precision: {macro_avg_precision}")
print(f"Macro Avg Recall: {macro_avg_recall}")
print(f"Macro Avg F1-Score: {macro_avg_f1}")
print(f"Weighted Avg Precision: {weighted_avg_precision}")
print(f"Weighted Avg Recall: {weighted_avg_recall}")
print(f"Weighted Avg F1-Score: {weighted_avg_f1}")

# Calculate ROC-AUC score
y_pred_proba = []

for i in range(len(test_gen)):
    X, y = test_gen[i]
    predictions = model.predict(X)
    y_pred_proba.extend(predictions[:, 1])

roc_auc = roc_auc_score(y_true, y_pred_proba)
print(f'ROC-AUC Score: {roc_auc}')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_frames(video_path, frame_size, frame_count):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, total_frames // frame_count)
    current_frame = 0

    while len(frames) < frame_count:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, frame_size)
        frame_array = np.array(frame_resized) / 255.0
        frames.append(frame_array)
        current_frame += frame_step

    cap.release()

    while len(frames) < frame_count:
        frames.append(frames[-1])

    return np.array(frames)

@app.route('/upload', methods=['POST'])
def upload_file():
   
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    incident_type = request.form.get('incidentType', 'Unknown Incident')
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        frames = extract_frames(filepath, frame_size, frame_count)
        frames = np.expand_dims(frames, axis=0)  # Add batch dimension
        prediction = model.predict(frames)
        predicted_class = np.argmax(prediction)
        
        if np.argmax(prediction) == 1:
            class_label = 'Incident Detected'
            pred=prediction[0][1]
            pred=(pred*100)
            pred=str(pred)
            test_acu=str(test_accuracy)
            true_pos=str(tp)
            true_neg=str(tn)
            false_pos=str(fp)
            false_neg=str(fn)
            precision_good = str(report_dict['Good']['precision'])
            recall_good = str(report_dict['Good']['recall'])
            f1_good = str(report_dict['Good']['f1-score'])
            support_good = str(report_dict['Good']['support'])
            precision_leaking = str(report_dict['Leaking']['precision'])
            recall_leaking = str(report_dict['Leaking']['recall'])
            f1_leaking = str(report_dict['Leaking']['f1-score'])
            support_leaking = str(report_dict['Leaking']['support'])
            accuracy = str(report_dict['accuracy'])
            roc=str(roc_auc)
            
            current_date = datetime.datetime.now().strftime("%d/%m/%Y")
            # Get the current time in UTC
            utc_time = datetime.datetime.now(timezone.utc)
            # Convert UTC time to IST
            ist_time = utc_time.astimezone(timezone(datetime.timedelta(hours=5, minutes=30))).strftime("%H:%M:%S")
            # Replace these with your actual Gmail credentials
            gmail_user = 'siriusanand1@gmail.com'
            gmail_password = 'spyl xfhj wkfy atbb'
            # Replace this with the email address of your supervisor
            supervisor_email = 'ramyame1510@gmail.com'
            # Create the email message
            msg = MIMEMultipart()
            msg['From'] = gmail_user
            msg['To'] = supervisor_email
            msg['Subject'] = 'Incident Alert!!'
            # Create the email body
            body = (f'An incident has been detected in the chemical industry.\n\n'
                f'Incident detected: {incident_type}\n'
                f'Date: {current_date}\n'
                f'Time: {ist_time} IST\n\n'              
                f'Please take immediate actions.')
            msg.attach(MIMEText(body, 'plain'))
            # Send the email
            try:
                server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
                server.ehlo()
                server.login(gmail_user, gmail_password)
                server.sendmail(gmail_user, supervisor_email, msg.as_string())
                server.quit()
                print('Email sent successfully!')
            except Exception as e:
                print('Error sending email:', e)
            
        else:
            class_label = 'No Incident Detected'
            pred=prediction[0][0]
            pred=(pred*100)
            pred=str(pred)
            test_acu=str(test_accuracy)
            true_pos=str(tp)
            true_neg=str(tn)
            false_pos=str(fp)
            false_neg=str(fn)
            precision_good = str(report_dict['Good']['precision'])
            recall_good = str(report_dict['Good']['recall'])
            f1_good = str(report_dict['Good']['f1-score'])
            support_good = str(report_dict['Good']['support'])
            precision_leaking = str(report_dict['Leaking']['precision'])
            recall_leaking = str(report_dict['Leaking']['recall'])
            f1_leaking = str(report_dict['Leaking']['f1-score'])
            support_leaking = str(report_dict['Leaking']['support'])
            accuracy = str(report_dict['accuracy'])
            roc=str(roc_auc)
        return jsonify({
        'classLabel': class_label,
        'pred': pred,
        "test_acu": test_acu,
        "true_pos": true_pos,
        "true_neg": true_neg,
        "false_pos": false_pos,
        "false_neg": false_neg,
        "precision_good": precision_good,
        "recall_good": recall_good,
        "f1_good": f1_good,
        "support_good": support_good,
        "precision_leaking": precision_leaking,
        "recall_leaking": recall_leaking,
        "f1_leaking": f1_leaking,
        "support_leaking": support_leaking,
        "accuracy": accuracy,
        "roc": roc,
        'incidentType': incident_type})

    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/static/<filename>', methods=['GET'])
def get_image(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath)
    return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
