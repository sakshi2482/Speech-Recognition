# -------------------------------
# Basic Speech Command Recognition
# Using DSP (MFCC features) + Simple Neural Network
# -------------------------------

import os
import numpy as np
import librosa        # for audio processing
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ---------------- PARAMETERS ----------------
SAMPLE_RATE = 16000    # samples per second
DURATION = 1           # keep 1 second of audio
N_MFCC = 13            # number of MFCC coefficients

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(file_path):
    """
    Load an audio file, convert it to 1 second,
    then extract MFCC features (basic DSP).
    """
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # Ensure all signals are of equal length (1 sec)
    if len(signal) > SAMPLE_RATE * DURATION:
        signal = signal[:SAMPLE_RATE * DURATION]
    else:
        pad_width = SAMPLE_RATE * DURATION - len(signal)
        signal = np.pad(signal, (0, pad_width), 'constant')

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)

    return mfcc.T   # shape = (time, mfcc)

# ---------------- DATASET PREPARATION ----------------
def prepare_dataset(data_dir, classes):
    """
    Read all .wav files from the dataset directory.
    Each folder name = class label (e.g., yes, no).
    """
    X, y = [], []
    for idx, label in enumerate(classes):
        folder = os.path.join(data_dir, label)
        for file in os.listdir(folder):
            if file.endswith('.wav'):
                path = os.path.join(folder, file)
                features = extract_features(path)
                X.append(features)
                y.append(idx)   # numeric label
    return np.array(X), np.array(y)

# ---------------- SIMPLE MODEL ----------------
def create_model(input_shape, num_classes):
    """
    A simple neural network for classification.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Flatten(),                    # flatten features
        tf.keras.layers.Dense(64, activation='relu'), # hidden layer
        tf.keras.layers.Dense(num_classes, activation='softmax') # output
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ---------------- MAIN ----------------
if __name__ == "__main__":
    # Define classes (same as folder names in dataset)
    classes = ['yes', 'no', 'up', 'down']   # change based on your dataset
    data_dir = r"C:\python\SpeechRecognitionProject\Large Dataset"  # dataset path

    print("Preparing dataset...")
    X, y = prepare_dataset(data_dir, classes)

    # Pad sequences so all features are same length
    max_len = max([x.shape[0] for x in X])
    X_padded = np.array([
        np.pad(x, ((0, max_len - x.shape[0]), (0, 0)), mode='constant')
        for x in X
    ])

    # Split into training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_padded, y, test_size=0.2, random_state=42
    )

    print("Creating model...")
    input_shape = X_train.shape[1:]   # (timesteps, features)
    model = create_model(input_shape, len(classes))
    model.summary()

    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=10, batch_size=32,
        validation_data=(X_val, y_val)
    )

    # Plot training vs validation accuracy
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Save trained model
    model.save("speech_model_basic.h5")

    # ---------------- TESTING ON NEW FILE ----------------
    def predict_command(file_path):
        features = extract_features(file_path)
        features_padded = np.pad(
            features, ((0, max_len - features.shape[0]), (0, 0)), mode='constant'
        )
        features_padded = np.expand_dims(features_padded, axis=0)  # add batch
        prediction = model.predict(features_padded).argmax(axis=1)[0]
        return classes[prediction]

    # Example test file
    test_file = r"C:\python\SpeechRecognitionProject\yes\sample_test.wav"
    print("Prediction for test file:", predict_command(test_file))
