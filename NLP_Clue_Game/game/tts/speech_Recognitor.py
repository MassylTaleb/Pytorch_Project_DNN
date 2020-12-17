import speech_recognition as sr
import getch

class Speech_Recognitor:

    def __init__(self):
        self.sr = sr
        self.r = sr.Recognizer()

    def poser_question(self, question):

        r = self.r
        sr = self.sr

        val = ''
        confirmation = 'non'
        
        while (confirmation != 'oui'):

            print("Utiliser le microphone.")
            
            with sr.Microphone() as source:
                print(question)
                audio = r.listen(source)
                # recognize speech using Google Speech Recognition
                try:
                    # for testing purposes, we're just using the default API key
                    # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
                    # instead of `r.recognize_google(audio)`
                    val = r.recognize_google(audio, language="fr-FR")

                    # with sr.Microphone() as source:
                    #     confirmation = r.listen(source)
                    #     confirmation = r.recognize_google(confirmation, language="fr-FR")
                    print(f"Avez vous dit (1:oui , 2:non): {val}")
                    key = getch.getch()
                    if key == "1":
                        confirmation = "oui"
                    elif key == "2":
                        print(f'Ok que vouliez-vous dire ?: {val}')
                        confirmation = "non"

                except sr.UnknownValueError:
                    print("Google Speech Recognition could not understand audio")
                except sr.RequestError as e:
                    print(
                        "Could not request results from Google Speech Recognition service; {0}".format(e))
            
        return val


if __name__ == "__main__":
    sr = Speech_Recognitor()
    var = sr.poser_question("dit de quoi big")
    print(var)
