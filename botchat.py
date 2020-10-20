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
     
# Creating a dictionary that maps each word to its number of occurences
word_count = {}
for question in clean_ques:
    for word in question.split():
        if word not in word_count:
            word_count[word] = 1
        else:
            word_count[word] += 1
for answer in clean_answ:
    for word in answer.split():
        if word not in word_count:
            word_count[word] = 1
        else:
            word_count[word] += 1

# Creating two dictionaries that map the questions and answers words to a unique integer
barrier = 20
ques_words_int = {}
word_number = 0
for word, count in word_count.items():
    if count >= barrier:
        ques_words_int[word] = word_number
        word_number += 1
answ_words_int = {}
word_number = 0
for word, count in word_count.items():
    if count >= barrier:
        answ_words_int[word] = word_number
        word_number += 1        
                        
# Adding the last tokens to these two dictionaries 
list_tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in list_tokens:
    ques_words_int[token] = len(ques_words_int) + 1
for token in list_tokens:
    answ_words_int[token] = len(answ_words_int) + 1

# Creating the inverse Dictionary of the answer dictionary
answer_int_word = {w_i: w for w, w_i in answ_words_int.items()} 

# Adding the End of String token to the end of every answer 
for i in range(len(clean_answ)):
    clean_answ[i] += ' <EOS>' 

# Translating all the questions and the answers into integers
# and Replacing all the words that were filtered out by <OUT> 
questions_to_int = []
for question in clean_ques:
    ints = []
    for word in question.split():
        if word not in ques_words_int:
            ints.append(ques_words_int[ '<OUT>'])
        else:
            ints.append(ques_words_int[word])
    questions_to_int.append(ints)   
answer_to_int = []
for answer in clean_answ:
    ints = []
    for word in answer.split():
        if word not in answ_words_int:
            ints.append(answ_words_int[ '<OUT>'])
        else:
            ints.append(answ_words_int[word])
    answer_to_int.append(ints)  

# Sorting questions and answers by the length of questions
sorted_clean_ques = []
sorted_clean_answ = []
for length in range(1, 25 + 1):
    for i in enumerate(questions_to_int):
        if len(i[1]) == length:
            sorted_clean_ques.append(questions_to_int[i[0]])
            sorted_clean_answ.append(answer_to_int[i[0]])

### Part 2 - Build the SEQ2SEQ Model
            
# Creating Placeholders for the inputs and the targets
def model_input():
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input')
    target = tf.placeholder(tf.int32, [None, None], name = 'target')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    return inputs, target, lr, keep_prob

# Preprocessing the targets
def preprocess_target(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    return preprocessed_targets

# Creating the Encoder RNN Layer
def encoder_rnn_layer(rnn_input, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                       cell_bw = encoder_cell,
                                                       sequence_length = sequence_length,
                                                       inputs = rnn_input,
                                                       dtype = tf.float32)
    return encoder_state
                                            
# Decoding the training set
def decode_training_Set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_fuction, keep_prob, batch_size):
    attention_state = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_state, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
    training_decod_fun = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                       attention_keys,
                                                                       attention_values,
                                                                       attention_score_function,
                                                                       attention_construct_function,
                                                                       name = "attn_dec_train")
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decod_fun,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_fuction(decoder_output_dropout)

# Decoding the test/validation set
def decode_validation_Set(encoder_state, decoder_cell, decoder_embedded_matrix, sos_id, eos_id, max_length, num_words, sequence_length, decoding_scope, output_fuction, keep_prob, batch_size):
    attention_state = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_state, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
    validation_decod_fun = tf.contrib.seq2seq.attention_decoder_fn_inference(output_fuction,
                                                                             encoder_state[0],
                                                                             attention_keys,
                                                                             attention_values,
                                                                             attention_score_function,
                                                                             attention_construct_function,
                                                                             decoder_embedded_matrix, 
                                                                             sos_id, 
                                                                             eos_id, 
                                                                             max_length, 
                                                                             num_words,
                                                                             name = "attn_dec_info")
    test_prediction, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                               validation_decod_fun,
                                                                                                               scope = decoding_scope)

    return test_prediction

# Creating the Decoder RNN(Recurrent neural network)
def decode_rnn(decoder_embedded_input, decoder_embedded_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weight = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        output_fuction = lambda x: 

     
    
    
        
    
    
    
    
    