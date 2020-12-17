__author__ = 'sylvieratte1'

import nltk
from nltk import load_parser
from aima.logic import *

def example1(s1):
    ep = load_parser('635/sem01app.fcfg')
    for tree in ep.parse(s1.split()):
        tree.draw()

#example1('Jean aime Marie')

def example2(s1):
    ep = load_parser('635/sem02app.fcfg')
    for tree in ep.parse(s1.split()):
        tree.draw()

#example2('Jean tua Marie avec une corde')

def example3(s1):
    ep = load_parser('635/sem03app.fcfg')
    for tree in ep.parse(s1.split()):
        tree.draw()

#example3('Jean tua Marie avec une corde dans la cuisine')

def example4(s1):
    ep = load_parser('635/sem04.fcfg')
    for tree in ep.parse(s1.split()):
        tree.draw()

#example4('Jean tua Marie avec une corde dans la cuisine')

def example5(s1):
    ep = load_parser('635/quant01.fcfg')
    for tree in ep.parse(s1.split()):
        tree.draw()

#example5('tous les cochons sont gros')

sents1 = ['La victime a été tuée à 22h']
sents2 = ['Jean tua Marie avec une corde']
sents3 = ['une corde tua Marie avec une corde dans la cuisine']
sents4 = ['tous les cochons sont gros']

def printResults (results):
    for result in results:
        for (synrep, semrep) in result:
            print(semrep)

printResults(nltk.interpret_sents(sents1, 'heure_crime.fcfg'))
#printResults(nltk.interpret_sents(sents2, '635/sem02app.fcfg'))
#printResults(nltk.interpret_sents(sents3, '635/sem03app.fcfg'))
#printResults(nltk.interpret_sents(sents3, '635/sem04.fcfg'))
#printResults(nltk.interpret_sents(sents4, '635/quant01.fcfg'))

