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
list_conversations = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    list_conversations.append(_conversation.split(','))
        
# Getting perfectly the questions and the answers
questions = []
answers = []   
for conver in list_conversations:
    for i in range(len(conver) - 1):
        questions.append(dic_line[conver[i]])
        answers.append(dic_line[conver[i+1]])

# Starting cleaning the texts
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#;:<>{}+=~?,]", "", text)
    return text

# Cleaning the questions
clean_ques = []
for question in questions:
    clean_ques.append(clean_text(question))
    
# Cleaning the answers
clean_answ = []
for answer in answers:
    clean_answ.append(clean_text(answer))
    
    
    
    
    
    
    
    
    
    