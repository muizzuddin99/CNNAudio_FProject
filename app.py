import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st
import sounddevice as sd
import soundfile as sf
from keras.preprocessing import image
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.models import Model
from keras.layers import Input, Flatten, Dense
from sklearn.metrics import classification_report

# Define the class labels
class_labels = ['background', 'scream', 'engine', 'storm', 'gunshot']

# Load the base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Define and load the trained model
inputs = Input(shape=base_model.output_shape[1:])
x = Flatten()(inputs)
x = Dense(1024, activation='relu')(x)
outputs = Dense(5, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Update this path to where your model weights are saved
model_weights_path = "C:/Users/ACER/Desktop/FYP/cnn_model_02.h5"
if os.path.exists(model_weights_path):
    model.load_weights(model_weights_path)
else:
    st.error(f"Model weights file not found at {model_weights_path}")

# Helper function to create a spectrogram
def create_spectrogram(y, sr, image_file):
    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr)
    fig.savefig(image_file)
    plt.close(fig)

# Helper function to record audio
def record_audio(duration, fs):
    st.write("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    st.write("Recording complete")
    return audio.flatten()

# Streamlit app
st.title('Home Safety Sound Detection')

# Create specific pages for each input mode
page = st.sidebar.selectbox("Choose input mode:", ['Single File', 'Multiple Files', 'Real-time Recording'])

if page == 'Single File':
    st.header('Single File Mode')
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])
    if uploaded_file is not None:
        file_names = [uploaded_file.name]
        audio_data = uploaded_file.read()
        with open("temp.wav", "wb") as f:
            f.write(audio_data)

        st.audio(audio_data, format='audio/wav', start_time=0)

        # Load the audio file
        y, sr = librosa.load("temp.wav", sr=None)

        # Create spectrogram
        create_spectrogram(y, sr, "temp.png")

        # Load the spectrogram image
        img = image.load_img("temp.png", target_size=(224, 224))
        st.image(img, caption=f'Spectrogram of {file_names[0]}', use_column_width=True)

        # Preprocess the image for prediction
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Extract features and make predictions
        features = base_model.predict(x)
        predictions = model.predict(features)

        # Display the prediction results
        st.write("Prediction Probabilities:")
        for i, label in enumerate(class_labels):
            st.write(f'{label}: {predictions[0][i]}')

elif page == 'Multiple Files':
    st.header('Multiple Files Mode')
    uploaded_files = st.file_uploader("Upload audio files", type=["wav"], accept_multiple_files=True)
    if uploaded_files is not None:
        file_names = [file.name for file in uploaded_files]
        predictions_dict = {}

        y_true = []  # Placeholder for true labels
        y_pred = []  # Placeholder for predicted labels

        for uploaded_file, file_name in zip(uploaded_files, file_names):
            audio_data = uploaded_file.read()
            with open("temp.wav", "wb") as f:
                f.write(audio_data)

            st.audio(audio_data, format='audio/wav', start_time=0)

            # Load the audio file
            y, sr = librosa.load("temp.wav", sr=None)

            # Create spectrogram
            create_spectrogram(y, sr, "temp.png")

            # Load the spectrogram image
            img = image.load_img("temp.png", target_size=(224, 224))
            st.image(img, caption=f'Spectrogram of {file_name}', use_column_width=True)

            # Preprocess the image for prediction
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            # Extract features and make predictions
            features = base_model.predict(x)
            predictions = model.predict(features)

            # Store prediction results
            predictions_dict[file_name] = predictions[0]
            y_pred.append(np.argmax(predictions[0]))

            # Dummy true label (replace with actual true label in a real scenario)
            y_true.append(np.random.randint(0, 5))

        # Display the prediction results
        st.write("Prediction Probabilities:")
        for file_name, prediction in predictions_dict.items():
            st.write(f"File: {file_name}")
            for i, label in enumerate(class_labels):
                st.write(f'{label}: {prediction[i]}')

        # Display classification report
        if len(y_true) > 0 and len(y_true) == len(y_pred):
            report = classification_report(y_true, y_pred, labels=[0, 1, 2, 3, 4], target_names=class_labels, output_dict=True)
            st.write("Classification Report:")
            st.json(report)
        else:
            st.write("Error: The number of true labels does not match the number of predictions.")

elif page == 'Real-time Recording':
    st.header('Real-time Recording Mode')
    duration = st.number_input("Enter duration for recording (seconds):", min_value=1, max_value=10, value=5)
    fs = 44100  # Sample rate
    if st.button("Start Recording"):
        audio = record_audio(duration, fs)
        sf.write("temp.wav", audio, fs)
        
        # Replay the recorded audio
        st.audio(audio, format='audio/wav', start_time=0, sample_rate=fs)

        # Load the audio file
        y, sr = librosa.load("temp.wav", sr=None)

        # Create spectrogram
        create_spectrogram(y, sr, "temp.png")

        # Load the spectrogram image
        img = image.load_img("temp.png", target_size=(224, 224))
        st.image(img, caption='Spectrogram of Real-time Recording', use_column_width=True)

        # Preprocess the image for prediction
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Extract features and make predictions
        features = base_model.predict(x)
        predictions = model.predict(features)
        y_pred = [np.argmax(predictions[0])]

        # Dummy true label (replace with actual true label in a real scenario)
        y_true = [np.random.randint(0, 5)]

        # Display the prediction results
        st.write("Prediction Probabilities:")
        for i, label in enumerate(class_labels):
            st.write(f'{label}: {predictions[0][i]}')

        # Display classification report
        if len(y_true) == len(y_pred):
            report = classification_report(y_true, y_pred, labels=[0, 1, 2, 3, 4], target_names=class_labels, output_dict=True)
            st.write("Classification Report:")
            st.json(report)
        else:
            st.write("Error: The number of true labels does not match the number of predictions.")

# Clean up temporary files
if os.path.exists("temp.wav"):
    os.remove("temp.wav")
if os.path.exists("temp.png"):
    os.remove("temp.png")
