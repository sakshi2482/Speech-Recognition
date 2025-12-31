import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
DATASET_PATH = r"C:\python\SpeechRecognitionProject\fivewordvocab"
CLASSES = ["house", "seven", "visual", "bed", "down", "cat", "one"]
X = []  
y = [] 
print("Loading dataset ")
for label in CLASSES:
    folder = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}")
        continue
    for file in os.listdir(folder)[:200]:
        if file.endswith(".wav"):
            file_path = os.path.join(folder, file)
            signal, sr = librosa.load(file_path, sr=16000)
            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=20)
            mfcc = np.mean(mfcc.T, axis=0) 
            X.append(mfcc)
            y.append(label)
X = np.array(X)
y = np.array(y)
print(f"✅ Dataset ready: {X.shape[0]} samples, {X.shape[1]} features each.")
encoder = LabelBinarizer()
y_encoded = encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(CLASSES), activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
print("Training model...")
history = model.fit(
    X_train, y_train,
    epochs=15, batch_size=32,
    validation_data=(X_test, y_test)
)
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {acc*100:.2f}%")
def predict_command(file_path):
    signal, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=20)
    mfcc = np.mean(mfcc.T, axis=0)
    mfcc = mfcc.reshape(1, -1)
    prediction = model.predict(mfcc, verbose=0)
    predicted_label = encoder.inverse_transform(prediction)[0]
    return predicted_label
TEST_PATH = os.path.join(DATASET_PATH, "test") 
if not os.path.exists(TEST_PATH):
    os.makedirs(TEST_PATH)
print("\n Please paste a .wav file into this folder for testing:")
print(TEST_PATH)
test_files = [f for f in os.listdir(TEST_PATH) if f.endswith(".wav")]
if len(test_files) == 0:
    print("No .wav file found in test folder. Paste a file and rerun the script.")
else:
    test_file = os.path.join(TEST_PATH, test_files[0])
    print(f"Testing on: {test_file}")
    predicted_word = predict_command(test_file)
    print(f" Predicted word: {predicted_word}")
