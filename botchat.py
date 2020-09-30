# Building a Intelkigent chatbot using Deep Nlp

# Importing the necessary libraries 
import numpy as np
import tensorflow as tf
import re 
import time

# Importing the dataset
lines = open('movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversations = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')

# Creating a dictionary that maps each line and its id
dic_line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        dic_line[_line[0]] = _line[4]
        
# Creating a list of all of the conversation
list_conver = []
for conv in conversations[:-1]:
    _conv = conversations.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace (" ", "")
    list_conver.append(_conv.str.split(','))
    
        
        