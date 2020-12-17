# Google Text to Speech API
from gtts import gTTS

# Library to play an mp3 using python
from playsound import playsound
import os

class Tts():
    def __init__(self):
        self.speech_path = os.path.dirname(os.path.abspath(__file__))+ '/speech.mp3'

    def speech(self, text):
        print(text)
        txt = gTTS(text=text, lang='fr')
        txt.save(self.speech_path)
        playsound(self.speech_path, True)
