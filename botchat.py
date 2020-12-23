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
        output_fuction = lambda x: tf.contrib.layers.fully_connected(x,
                                                                     num_words,
                                                                     None,
                                                                     scope = decoding_scope,
                                                                     weights_intializers = weight,
                                                                     biases_initializer = biases)
        training_predictions = decode_training_Set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   decoding_scope,
                                                   output_fuction,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_validation_Set(encoder_state,
                                                 decoder_cell,
                                                 decoder_embedded_matrix,
                                                 word2int['<SOS>'],
                                                 word2int['<EOS>'],
                                                 sequence_length - 1,
                                                 num_words,
                                                 decoding_scope,
                                                 output_fuction,
                                                 keep_prob,
                                                 batch_size)
           
    return training_predictions, test_predictions

# Building the seq2seq model
def seq2seq_model(inputs, 
                  targets, 
                  keep_prob, 
                  batch_size, 
                  sequence_length, 
                  answer_num_word, 
                  question_num_word,
                  encoder_embedding_size,
                  decoder_embedding_size,
                  rnn_size,
                  num_layers,
                  questions_to_int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answer_num_word + 1,
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn_layer(encoder_embedded_input,
                                      rnn_size,
                                      num_layers,
                                      keep_prob,
                                      sequence_length)
    preprocessed_targets = preprocess_target(targets, ques_words_int, batch_size)
    decoder_embedded_matrix = tf.Variable(tf.random_uniform([question_num_word + 1, 
                                                            decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embedded_matrix, preprocessed_targets)
    training_predictions, test_predictions = decode_rnn(decoder_embedded_input,
                                                        decoder_embedded_matrix,
                                                        encoder_state,
                                                        question_num_word,
                                                        sequence_length,
                                                        rnn_size,
                                                        num_layers,
                                                        ques_words_int,
                                                        keep_prob,
                                                        batch_size)
    return training_predictions, test_predictions
    

## Part 3- Training SEQ2SEQ Model

# Setting the Hyperparameters
epochs = 100 
batch_size = 64
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5

# Define a session
tf.compat.v1.reset_default_graph()
session = tf.compat.v1.InteractiveSession()

# Loading the model inputs
inputs, targets, lr, keep_prob = model_input()

# Setting the sequence length 
sequence_length = tf.compat.v1.placeholder_with_default(25, None, name ='sequence_length')

# Getting the shape of the inputs sensor 
input_shape  = tf.shape(inputs)

# Getting the training and test predictions 
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs),
                                                       [-1],
                                                       targets,
                                                       keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(answer_to_int),
                                                       len(questions_to_int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       ques_words_int)

# Setting up the Loss Error, The Optimizer and gradient Clipping
with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                  targets,
                                                  tf.ones([input_shape[0],sequence_length]))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable)for grad_tensor, grad_variable in gradients if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)


# Padding the Sequences with the <PAD> TOKEN
def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence))for sequence in batch_of_sequences]


# Splitting the data into batches of questions and answers
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions) // batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index : start_index + batch_size]
        answers_in_batch = answers[start_index : start_index + batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, ques_words_int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answ_words_int))
        yield padded_questions_in_batch, padded_answers_in_batch
        

# Splitting the questions and answers into training and validation set
training_validation_split = int(len(sorted_clean_ques) * 0.15)
training_questions = sorted_clean_ques[training_validation_split:]
training_answers = sorted_clean_answ[training_validation_split:] 
validation_questions = sorted_clean_ques[:training_validation_split]
validation_answers = sorted_clean_answ[:training_validation_split]    


# Training 
batch_index_check_training_loss = 100
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) - 1
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 1000
checkpoint = "chatbot_weights.ckpt"
session.run(tf.global_variables_initializer())
for epoch in range(1, epochs + 1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs: padded_questions_in_batch,
                                                                                               targets: padded_answers_in_batch,
                                                                                               lr: learning_rate,                                      
                                                                                               sequence_length: padded_answers_in_batch.shape[1],
                                                                                               keep_prob: keep_probability})
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds'.format(epoch,
                                                                                                                                       epochs,
                                                                                                                                       batch_index,
                                                                                                                                       len(training_questions)  // batch_size,
                                                                                                                                       total_training_loss_error / batch_index_check_training_loss,
                                                                                                                                       int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
        if batch_size % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
                batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch,
                                                                       targets: padded_answers_in_batch,
                                                                       lr: learning_rate,
                                                                       sequence_length: padded_answers_in_batch.shape[1],
                                                                       keep_prob: 1})
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)
            print('validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print('I can speak better right now.')
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("Sorry i can not speak better, I need more practice.")
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    if early_stopping_check == early_stopping_stop:
        print("My apologies, I can not speak more btter than that.")
        break
print("Its Done")    
            

# Part4 Testing the seq2seq Model

# Loading the weights and Running the session
checkpoint = "./chatbot_weights.ckpt"
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(session, checkpoint)

# Converting the questions from strings to lists of encoding integers
def convert_string2int(question, word2int):
    question = clean_text(question)
    return [word2int.get(word, word2int['<OUT>']) for word in question.split()]


                
            
            
                                                                           
                                                                           
                                                                            


  
        
        
    
    
    
    
    
                                                          
    




    
    
    
    
    
    
    
    
    
    
        

     
    
    
        
    
    
    
    
    