import pytest
import logging
from emo_speech_recognizer import EmoSpeechRecognizer

logging.basicConfig(level=logging.INFO)


@pytest.fixture(scope="session")
def emo_speech_model(request):
    # Instantiate the speech recognition model
    model = EmoSpeechRecognizer()
    yield model
