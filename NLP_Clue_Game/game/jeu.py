# Google Text to Speech API
from gtts import gTTS
# Library to play an mp3 using python
from playsound import playsound
import getch

from .tts.tts import Tts
from .agent import Agent
import json
import os, sys


class Jeu:
    def __init__(self):
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.position = 0
        self.tts = Tts()

        with open(self.path+"/jeu.json", "r") as json_file:
            jeu = json.load(json_file)
            self.map = jeu["map"]
            self.personnes = jeu['personnes']
            self.armes = jeu['armes']

        with open(self.path+"/room_facts.json", "r") as json_file:
            room_facts = json.load(json_file)
            self.room_facts = room_facts

        # self.agent = Agent(self)

    def play(self):
        
        self.print_info_jeu()
        
        # Instance d'Agent
        agent = Agent(self)

        with open(self.path+"/initial_facts.json", "r") as json_file:
            facts = json.load(json_file)

        agent.personne_facts(facts['personne_vivantes'])
        agent.personne_marques_fact(facts['personne_marque'])
        agent.arme_marque_facts(facts['arme_marque'])


        agent.presentation()

        while True:

            print("[JEU] jouer une tour !!")
            # in_val = input("Press any key :  ")
            communication_f = self.path + '/communication.txt'
            in_val = getch.getch()

            if(in_val == "a"):
                print("[JEU] Piece Précédente")
                print('\n')

                agent.prev_Piece()

            elif(in_val == 'd'):
                print("[JEU] Piece Suivante")
                print('\n')
                agent.next_Piece()

            elif(in_val == "w"):
                print("[JEU] Première Pièce")
                print('\n')
                agent.firt_Piece()

            elif(in_val == "s"):
                print("[JEU] Dernière Pièce")
                print('\n')
                agent.last_Piece()

            elif(in_val == "p"):
                print("[JEU] Regle d'inference")
                print('\n')
                agent.printKB()

            elif(in_val == "c"):
                print("[JEU] CONCLUSION")
                print('\n')
                agent.getConclusion()

            elif(in_val == "4"):
                self.tts.speech("\nExiting game !!!")
                exit()

            else:
                text = "Wrong Input"


    def print_info_jeu(self):
        print("Key : Commande")
        print(
        '  1 : Oui'
        '\n  2 : Non'
        '\n  w : Permier Pièce'
        '\n  s : Dernière Pièce'
        '\n  a : Pièce précédente'
        '\n  d : Pièce suivante'
        '\n  p : print les clauses'
        '\n  c : print la decution du detective')