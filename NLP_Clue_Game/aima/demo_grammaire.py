__author__ = 'sylvieratte1'

import nltk

from nltk import load_parser

def example1(s1):
    grammar1 = nltk.data.load('635/syn01.cfg')
    ep = nltk.EarleyChartParser(grammar1)
    for tree in ep.parse(s1.split()):
        tree.draw()

#example1('le Colonel_Mustard a_tue Madame_Peacock dans le salon avec la corde')

def example2(s1):
    ep = load_parser('635/feat0.fcfg')
    for tree in ep.parse(s1.split()):
        tree.draw()

#example2('Jody likes girls')
#example2('Marie aime les chats')

def example3(s1):
    ep = load_parser('635/simple-sem.fcfg')
    for tree in ep.parse(s1.split()):
        tree.draw()

#example3('all dogs bark')

def example4(s1):
    ep = load_parser('635/event.fcfg')
    for tree in ep.parse(s1.split()):
        tree.draw()

#example4('Cyril gives a dog to a boy')

def example5(s1):
    ep = load_parser('635/rattachement.cfg')
    for tree in ep.parse(s1.split()):
        tree.draw()

example5('J ai_vu un homme dans le parc avec un telescope')

def example6(s1):
    ep = load_parser('635/rattachement2.cfg')
    for tree in ep.parse(s1.split()):
        print(tree)

#example6('il a_vu l ours sur la montagne pendant la nuit')