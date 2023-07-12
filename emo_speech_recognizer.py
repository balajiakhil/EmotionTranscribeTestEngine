""" Emotion Speech Recognizer """
import os
import sys
import warnings
import json
import pickle
import vosk
from pydub import AudioSegment
from keras.models import model_from_json
from keras.optimizers import RMSprop
import pandas as pd
import numpy as np
import librosa
import librosa.display


# Ignore warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


class EmoSpeechRecognizer:
    """
        EmoSpeechRecognizer class perform Emotion & Speech recognition.
    """
    def __init__(self, target_sample_rate: int = 16000):
        """
        Initialize the EmoSpeechRecognizer object
        :param target_sample_rate:
        """
        self.vosk_model = vosk.Model(os.path.join("SpeechModel", "vosk-model-en-us-0.22"))
        self.target_sample_rate = target_sample_rate

    def transform_dataset(self, file_path):
        data, sample_rate = librosa.load(file_path, res_type='kaiser_fast', duration=2.5, sr=44100, offset=0.5)
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13), axis=0)
        newdf = pd.DataFrame(data=mfccs).T
        newdf = np.expand_dims(newdf, axis=2)
        return newdf

    def load_emotion_model_and_predict(self, file_path):
        # Load the emotion recognition model architecture
        emo_json = os.path.join("EmotionModel", "emotion_recognition_model_json.json")
        with open(emo_json, 'r', encoding='utf-8') as fname:
            loaded_model_json = fname.read()
        loaded_model = model_from_json(loaded_model_json)

        # Load the emotion recognition model weights
        loaded_model.load_weights(os.path.join("EmotionModel", "emotion_recognition_model.h5"))

        # Define the optimizer
        opt = RMSprop(learning_rate=0.00001)
        loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        newdf = self.transform_dataset(file_path)
        newpred = loaded_model.predict(newdf, batch_size=16, verbose=2)

        # Load the label encoder
        with open(os.path.join("EmotionModel", "labels"), 'rb') as infile:
            lb = pickle.load(infile)

        # Get the final predicted label
        final = newpred.argmax(axis=1)
        final = final.astype(int).flatten()
        final = lb.inverse_transform(final)
        # final = str(final.tostring())

        return final

    def recognize_speech(self, audio_file: str, emotion: str = None) -> str:
        try:
            # Load the audio file
            audio = AudioSegment.from_file(audio_file)

            preprocessing_actions = {
                "angry": lambda audio: audio + 10,  # Increase volume by 10 dB
                "sad": lambda audio: audio - 10,  # Decrease volume by 10 dB
                "happy": lambda audio: audio + 5,  # Increase volume by 5 dB
                "fear": lambda audio: audio.high_pass_filter(2000),  # Apply a high-pass filter
                "disgust": lambda audio: audio.low_pass_filter(2000),
            }

            # Apply emotion-based preprocessing techniques
            for keyword, preprocess_action in preprocessing_actions.items():
                if emotion and keyword in emotion.lower():
                    audio = preprocess_action(audio)
                    break

            # Convert audio to the target sample rate
            audio = audio.set_frame_rate(self.target_sample_rate)

            # Extract audio data as bytes
            audio_data = audio.raw_data

            # Perform speech recognition using the initialized Vosk model
            rec = vosk.KaldiRecognizer(self.vosk_model, self.target_sample_rate)
            rec.AcceptWaveform(audio_data)

            # Get the recognized text from the result_data
            result_data = json.loads(rec.Result())
            return result_data.get("text", "")

        except Exception as error:
            # Handle audio processing or speech recognition error
            print(f"Error: {str(error)}")
            return ""

    def emo_speech_recognizer(self, audio_path):
        # Load emotion recognition model and predict emotion
        emotion_result = self.load_emotion_model_and_predict(audio_path)

        # Recognize speech with emotion
        speech_result = self.recognize_speech(audio_path, emotion_result[0])

        return emotion_result, speech_result
