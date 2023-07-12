import os
import csv
import logging
import pytest
from emo_speech_recognizer import EmoSpeechRecognizer

# Configure logging
logging.basicConfig(level=logging.INFO)


def csv_data():
    data = []
    file_path = os.path.join("dataset_path.csv")

    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            audio_file = row[0]
            emotion = row[1]
            transcribe = row[2]
            data.append((os.path.join("Audio", audio_file), emotion, transcribe))

    return data


@pytest.mark.parametrize("audio_file, emotion, expected_transcription", csv_data())
class TestSpeechRecognition:
    def test_transcribe_audio(self, audio_file, emotion, expected_transcription):
        logging.info(
            f"Running test: audio_file={audio_file}, emotion={emotion}, expected_transcription={expected_transcription}")

        # Instantiate the speech recognition model
        emo_speech_model = EmoSpeechRecognizer()

        # Perform speech and Emotion recognition
        emotion_result, speech_result = emo_speech_model.emo_speech_recognizer(audio_file)

        logging.info(f"Emotion result: {emotion_result}")
        logging.info(f"Speech result: {speech_result}")

        # Assert Speech
        assert speech_result == expected_transcription, \
            f"Failed to recognize Transcription, actual:{expected_transcription} observed: {speech_result}"

        # assert emotion
        assert emotion == emotion_result, \
            f"Failed to recognize Emotion, actual:{emotion} observed: {emotion_result}"
