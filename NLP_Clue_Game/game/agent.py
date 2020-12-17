
# Google Text to Speech API
from gtts import gTTS

# Library to play an mp3 using python
from playsound import playsound
import os
from .inference_engine import CrimeInference
from .tts.tts import Tts
from .tts.speech_Recognitor import Speech_Recognitor
import json
from random import randrange
import getch

import nltk

class Agent:

    def __init__(self, jeu):
        self.jeu = jeu
        self.first_position = 0
        self.position = 0
        self.tts = Tts()
        self.sr = Speech_Recognitor()
        self.inference = CrimeInference(self.jeu.map, self.jeu.personnes, self.jeu.armes)
        self.trouver_mort = False
        self.personne_morte = ""




    def setFirstPosition(self, position):
        self.first_position = position


    def presentation(self):
        print('\n\n')
        self.askFirstPosition()
        self.inspect_Piece()

    
    def inspect_Piece(self):
        """
        Cette methode est la suite logique d'une inspection d'une piece
        """

        # Selectionner la position de la piece 
        piece = self.jeu.map[self.position]
        self.print_room_info(piece)

        room_fact = self.jeu.room_facts[piece]
        
        room_person = room_fact['personne']
        room_weapon = room_fact['arme']

        # Ajouter la clause arme_piece
        self.add_clause(room_weapon, "arme_piece")

        # Questionner la personne dans la piece
        self.questionner_pesonne_piece(room_person)

        # Questionner l'arme dans la piece
        self.questionner_heure_arme(room_weapon)

    



    def questionner_pesonne_piece(self, room_person):

        if(len(room_person) > 1):
            for p in room_person:
                self.add_clause(p,"personne_piece")
                personne = p[0].split()[0]
                self.setAlive(personne)
                self.questionner_personne_piece_mort(personne)

        else:
            self.add_clause(room_person,"personne_piece")
            personne = room_person[0].split()[0]
            self.setAlive(personne)
            self.questionner_heure_personne(personne)



    def questionner_personne_piece_mort(self, personne):

        if self.inference.is_Person_alive(personne) and self.trouver_mort == True:

            reponse_correct = False
            while (reponse_correct != True): 
                
                self.print(f"{personne}, connaissez-vous l'heure du décès? (1:oui , 2:non): ")

                in_val = getch.getch()

                # 1 = Oui
                # 2 = Non
                if (in_val == "1"):
                    # val = input("[AGENT] Je vous écoute : ")
                    val = self.random_question_input(f"D'accord a quelle heure {self.inference.get_victim()} est morte:")
                    # Ajoute l'heure du deces
                    self.add_clause([val], "personne_morte_heure")
                    # Ajout clause une heure apres
                    uneHeureApres = self.inference.get_crime_hour() + 1 
                    self.inference.add_clause('UneHeureApresCrime({})'.format(uneHeureApres))
                    self.questionner_heure_personne(personne)
                    reponse_correct = True
                elif(in_val == "2"):
                    self.tts.speech("D'accord, merci.")
                    reponse_correct = True
                else:
                    reponse_correct = False
        return


    def questionner_heure_personne(self, personne):

        if self.inference.is_Person_alive(personne) and self.trouver_mort == True:
            reponse_correct = False
            while (reponse_correct != True):
                # print(self.inference.get_crime_hour())
                heureapres = self.inference.get_crime_hour() + 1 
                # room = input(f"[AGENT] Où se trouvait {personne} une heure après le crime({heureapres})? (nom de la pièce) : ")
                room = self.random_question_input(f"[AGENT] Où se trouvait {personne} une heure après le crime({heureapres})? (nom de la pièce) : ")
                if(room.lower() in self.jeu.map):
                    personne_piece_heure = f"{personne} se trouvait dans la {room} à {heureapres}h"
                    self.add_clause([personne_piece_heure], "personne_piece_heure")
                    reponse_correct = True
                else:
                    reponse_correct = False
        return
       

    # Le fusil se trouve dans la cuisine
    def questionner_heure_arme(self, arme_sentence):
        
        arme = arme_sentence[0].split()[1]
        # arme = self.inference.get_arme_piece(piece)
        heureCrime = self.inference.get_crime_hour()
        uneHeureApres = self.inference.get_crime_hour_plus_one()

        arme_pieceInitial = ""
        piece_arme_uneheureApres = ""

        if self.mort_trouvee():

            reponse_correct = False
            while (reponse_correct != True):
                
                # arme_pieceInitial = input(f"[AGENT] Où se trouve le {arme} initialement?: ")
                arme_pieceInitial = self.random_question_input(f"[AGENT] Où se trouve le {arme} initialement?: ")

                if (arme_pieceInitial.lower() in self.jeu.map):
                    # Arme piece heure
                    arme_piece_heure = [f"Le {arme} était dans la {arme_pieceInitial} à {heureCrime}h"]
                    print(arme_piece_heure)
                    self.add_clause(arme_piece_heure, "arme_piece_heure")
                    reponse_correct = True
                else:
                    reponse_correct = False

            reponse_correct = False
            while (reponse_correct != True):

                # arme = self.inference.get_arme_piece(piece)
                heureCrime = self.inference.get_crime_hour()
                # piece_arme_uneheureApres = input(f"[AGENT] Où se trouvait le {arme} une heure après le crime?: ")
                piece_arme_uneheureApres = self.random_question_input(f"[AGENT] Où se trouve le {arme} une heure après le crime?: ")

                if (piece_arme_uneheureApres.lower() in self.jeu.map):
                    # Arme piece heure
                    arme_piece_heure = [f"Le {arme} était dans la {piece_arme_uneheureApres} à {uneHeureApres}h"]
                    print(arme_piece_heure)
                    self.add_clause(arme_piece_heure, "arme_piece_heure")
                    reponse_correct = True
                else:
                    reponse_correct = False
            
        return


    def poser_communication(self, question):
        reponse = ""
        
        print("Utiliser le fichier communication")
        print("Appuyer sur 1 pour lire la communication")
        
        key = getch.getch()
        if(key == "1"):
            f = open("/home/ziz/school/log635/LOG635_labo/Lab2/game/communication.txt")
            reponse = f.read()
            print(reponse)

        return reponse

    def askFirstPosition(self):

        # self.print("Pouvez-vous m'indiquer la première pièce?")
        # premiere_piece = input("Nom de la Pièce: ").lower()

        premiere_piece = self.random_question_input("[AGENT] Pouvez-vous m'indiquer qu'elle est la première pièce?").lower()

        if(premiere_piece in self.jeu.map):
            position = self.jeu.map.index(premiere_piece)
            self.setFirstPosition(position)
            self.setPosition(position)
        else:
            self.print("Je n'ai pas bien compris.")
            self.askFirstPosition()





    def random_question_input(self, question):
        """
        Cette methode permet de choisir alléatoirement le canal de communincation
        """
        
        print(question)

        rand_num = randrange(3)
        reponse = ""
        if '[AGENT]' in question :
            q = question.replace('[AGENT]','')
            self.tts.speech(q)
        else:
            self.tts.speech(question)
            pass
            
        if(rand_num == 0):
            reponse = self.poser_communication(question)
        elif(rand_num == 1):
            reponse = self.sr.poser_question(question)
        elif(rand_num == 2):
            reponse = input("Utiliser le terminal ici :")

        return reponse  

    def print_room_info(self, room_name_json):
        
        room_fact = self.jeu.room_facts[room_name_json]
        # self.tts.speech(f"J'entre dans {room_name_json}")

        print('-----------ROOM INFORMATION-----------')
        print(f'PIÈCE : {room_name_json}')

        if(len(room_fact["personne"]) > 1):
            for pers in room_fact["personne"]:
                print(f'PERSONNE : {pers}')
        else:
            print(f'PERSONNE : {room_fact["personne"]}')

        # self.tts.speech(room_weapon[0])
        print(f'ARME : {room_fact["arme"]}')
        print('--------------------------------------\n')


    def mort_trouvee(self):
        trouver = False
        if self.trouver_mort == True:
            trouver = True

        return trouver


    def poser_question(self, question_string):
        self.tts.speech(f"{question_string}")
        input("")

    def next_Piece(self):
        self.increase_position()
        self.inspect_Piece()

    def prev_Piece(self):
        self.decrease_position()
        self.inspect_Piece()

    def firt_Piece(self):
        self.setPosition(self.first_position)
        self.inspect_Piece()
    
    def last_Piece(self):
        self.setPosition(len(self.jeu.map))
        self.inspect_Piece()

    def increase_position(self):
        self.position = (self.position + 1) % len(self.jeu.map) 

    def decrease_position(self):
        if (self.position == 0):
            self.position = len(self.jeu.map) -1 

        else:
            self.position -= 1  

    def setPosition(self, position):
        self.position = position


    def printKB(self):
        for c in self.inference.crime_kb.clauses:
            print(c)

    def getConclusion(self):
        print("Pièce du crime : ", self.inference.get_crime_room())
        print("Arme du crime : ", self.inference.get_crime_weapon())
        print("Personne victime : ", self.inference.get_victim())
        print("Heure du crime : ", self.inference.get_crime_hour())
        print("Meurtrier : ", self.inference.get_meurtrier())
        print("Suspect : ", self.inference.get_suspect())
        print("Personnes innocentes : ", self.inference.get_innocent())
        print("Armes suspects : ", self.inference.get_arme_suspect())


    def personne_facts(self, personne_facts):
        self.add_clause(personne_facts[0], "personne_morte")
        personne_facts.pop(0)
        for pF in personne_facts:
            self.add_clause(pF, "personne_vivant")
    
    def personne_marques_fact(self, personne_marque):
        self.add_clause(personne_marque, "personne_marque")

    def arme_marque_facts(self, arme_marque):
        for aM in arme_marque:
            det_arme = [aM[0].split()[0] +  " " + aM[0].split()[1]]
            # print(det_arme)
            # self.add_clause(det_arme, "det_arme")
            self.add_clause(aM, "arme_marque")

    def arme_piece_heure_fact(self, arme_piece_heure):
        for aM in arme_piece_heure:
            self.add_clause(aM, "arme_piece_heure")

    

    def setAlive(self, personne):
        if self.inference.is_Person_alive(personne) == False:
            self.trouver_mort = True
        else:
            return



    def add_clause(self, fact, grammaire_name):
        self.inference.add_tofol_clause(fact, f'{self.jeu.path}/grammars/{grammaire_name}.fcfg'
)
    # def random_question_input(question, salle_mort=False):
    #     rand_val = randrange(3)
    #     if(salle_mort):


    def print(self, string):
        print(f'[AGENT] : {string}' )