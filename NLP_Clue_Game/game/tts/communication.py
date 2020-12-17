from tts import Tts
from speech_Recognitor import  Speech_Recognitor


class Communication:

    def __init__(self):
        self.tts = Tts()
        self.sp = Speech_Recognitor() 

    def poser_question(self, question):
        print("Je vous écoute : ")
        self.hello = "bonjore"

    def printHello(self):
        print(self.hello)