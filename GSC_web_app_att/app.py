import re
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import*
from tensorflow.keras.layers import *
from tensorflow import keras 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.backend import clear_session
import tensorflow as tf
#gpus = tf.config.list_physical_devices('GPU')
#gpu = gpus[0]

#tf.config.experimental.set_memory_growth(gpu, True)

import pickle
from flask import Flask, jsonify, request


# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)


###################################################
#Load  all these saved data

with open('tokenizer_enc.pickle', 'rb') as file:
    tokenizer_enc = pickle.load(file)
    
with open('tokenizer_dec.pickle', 'rb') as file:
    tokenizer_dec = pickle.load(file)
    
with open('vocab_size_enc.pickle', 'rb') as file:
    vocab_size_enc = pickle.load(file)

with open('vocab_size_dec.pickle', 'rb') as file:
    vocab_size_dec = pickle.load(file)
    
#Parameters

batch_size=32
lstm_size=128
max_dec = 50
max_enc = 29
embedding_dim = 100
dense_units = 256
att_units = 256
latent_dim=192
    

#Encoder class
class Encoder(tf.keras.layers.Layer):
    '''
    Encoder model -- That takes a input sequence and returns encoder-outputs,encoder_final_state_h,encoder_final_state_c
    '''

    def __init__(self,inp_vocab_size,embedding_size,lstm_size,input_length):
        super(Encoder, self).__init__()
        self.lstm_size = lstm_size
        #Initialize Embedding layer
        self.enc_embed = Embedding(input_dim = inp_vocab_size, output_dim = embedding_size)
        #Intialize Encoder LSTM layer
        self.enc_lstm = LSTM(lstm_size, return_sequences = True, return_state = True)
        
    def call(self,input_sequence,states):
        '''
          This function takes a sequence input and the initial states of the encoder.
          Pass the input_sequence input to the Embedding layer, Pass the embedding layer ouput to encoder_lstm
          returns -- encoder_output, last time step's hidden and cell state
        '''
        embedding = self.enc_embed(input_sequence)
        output_state, enc_h, enc_c = self.enc_lstm(embedding, initial_state = states)
        return output_state, enc_h, enc_c
    
    def initialize_states(self,batch_size):
      '''
      Given a batch size it will return intial hidden state and intial cell state.
      If batch size is 32- Hidden state is zeros of size [32,lstm_units], cell state zeros is of size [32,lstm_units]
      '''
      return [tf.zeros((batch_size, self.lstm_size)), tf.zeros((batch_size, self.lstm_size))]

    
    
#Attention class
class Attention(tf.keras.layers.Layer):

    def __init__(self,scoring_function, att_units):
        
        # Please go through the reference notebook and research paper to complete the scoring functions
        
        super(Attention, self).__init__()
        self.scoring_function = scoring_function
        
        # Intialize variables needed for Dot score function here
        if scoring_function == 'dot':
            self.dot = Dot(axes = (1, 2))
            pass
        
        # Intialize variables needed for General score function here
        if scoring_function == 'general':
            self.W = Dense(att_units)
            self.dot = Dot(axes = (1, 2))
            pass
        
        # Intialize variables needed for Concat score function here
        if scoring_function == 'concat':
            self.W1 = Dense(att_units)
            self.W2 = Dense(att_units)
            self.V = Dense(1)
            pass
        
    def call(self,decoder_hidden_state,encoder_output):
        
        '''
        Attention mechanism takes two inputs current step -- decoder_hidden_state and all the encoder_outputs.
        Based on the scoring function we will find the score or similarity between decoder_hidden_state and encoder_output.
        Multiply the score function with your encoder_outputs to get the context vector.
        Function returns context vector and attention weights(softmax - scores)
        '''
    
        decoder_hidden_state = tf.expand_dims(decoder_hidden_state, 1)
        
        if self.scoring_function == 'dot':
            # Implement Dot score function here
            score = tf.transpose(self.dot([tf.transpose(decoder_hidden_state, (0, 2, 1)), encoder_output]), (0, 2,1))
            pass
            
        elif self.scoring_function == 'general':
            # Implement General score function here
            mul = self.W(encoder_output)
            score = tf.transpose(self.dot([tf.transpose(decoder_hidden_state, (0, 2, 1)), mul]), (0, 2,1))
            pass
            
        elif self.scoring_function == 'concat':
            # Implement General score function here
            inter = self.W1(decoder_hidden_state) + self.W2(encoder_output)
            tan = tf.nn.tanh(inter)
            score = self.V(tan)
            pass
        
        attention_weights = tf.nn.softmax(score, axis =1)
        context_vector = attention_weights * encoder_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights
    

#One_Step_Decoder class
class One_Step_Decoder(tf.keras.Model):
    
    def __init__(self,tar_vocab_size, embedding_dim, input_length, dec_units ,score_fun ,att_units):
        
        # Initialize decoder embedding layer, LSTM and any other objects needed
        super().__init__()
        self.tar_vocab_size = tar_vocab_size
        self.embedding_dim = embedding_dim
        self.input_dim = input_length
        self.lstm_units = dec_units
        self.score_fun = score_fun
        self.att_units = att_units
        self.context_vector = 0
        self.attention_weights = 0
        self.dec_output = 0
        self.decoder_state_h = 0
        self.decoder_state_c = 0

        self.Embedding_layer = Embedding(input_dim= self.tar_vocab_size, output_dim= self.embedding_dim,input_length= self.input_dim,
                                      mask_zero = True, name = "decoder_embedding_layer")
        self.LSTM_layer = LSTM(units = self.lstm_units, return_sequences= True,return_state= True, name = "decoder_LSTM_layer")

        self.Attention_layer = Attention(self.score_fun, self.att_units)

        self.Dense_layer = Dense(units = self.tar_vocab_size)
        
    def call(self,input_to_decoder, encoder_output, state_h,state_c):
        '''
        One step decoder mechanisim step by step:
      A. Pass the input_to_decoder to the embedding layer and then get the output(batch_size,1,embedding_dim)
      B. Using the encoder_output and decoder hidden state, compute the context vector.
      C. Concat the context vector with the step A output
      D. Pass the Step-C output to LSTM/GRU and get the decoder output and states(hidden and cell state)
      E. Pass the decoder output to dense layer(vocab size) and store the result into output.
      F. Return the states from step D, output from Step E, attention weights from Step -B
        '''
        embedded_output = self.Embedding_layer(input_to_decoder)
        
        self.context_vector,self.attention_weights = self.Attention_layer(state_h,encoder_output)
        self.context_vector = tf.expand_dims(self.context_vector, axis = 1)
        
        concanated_decoder_input = tf.concat([self.context_vector,embedded_output], axis = -1)
        
        self.dec_output, self.decoder_state_h, self.decoder_state_c = self.LSTM_layer(concanated_decoder_input,
                                                                                      initial_state=[state_h, state_c])
        
        output = self.Dense_layer(self.dec_output)
        output = tf.squeeze(output, axis =1)
        
        self.context_vector = tf.squeeze(self.context_vector)
        
        return output, self.decoder_state_h, self.decoder_state_c, self.attention_weights,self.context_vector
    
#Decoder class
class Decoder(tf.keras.Model):
    def __init__(self,out_vocab_size, embedding_dim, input_length, dec_units ,score_fun ,att_units):
      #Intialize necessary variables and create an object from the class onestepdecoder
        
        super(Decoder, self).__init__()
        self.input_length = input_length
        self.out_vocab_size = out_vocab_size
        self.one_step_decoder = One_Step_Decoder(out_vocab_size, 
                                               embedding_dim, 
                                               input_length, 
                                               dec_units,
                                               score_fun,
                                               att_units)
        
        self.out_vocab_size = out_vocab_size
        
    def call(self, input_to_decoder, encoder_output, decoder_hidden_state, decoder_cell_state):
        
        #Initialize an empty Tensor array, that will store the outputs at each and every time step
        #Create a tensor array as shown in the reference notebook
        
        #Iterate till the length of the decoder input
            # Call onestepdecoder for each token in decoder_input
            # Store the output in tensorarray
        # Return the tensor array
        
        all_outputs = tf.TensorArray(dtype = tf.float32, size= input_to_decoder.shape[1])
        
        for timestep in range(input_to_decoder.shape[1]):
            output, decoder_hidden_state, decoder_cell_state, _, _ = self.one_step_decoder(input_to_decoder[:, timestep:timestep+1], 
                                                                                             encoder_output, 
                                                                                             decoder_hidden_state,
                                                                                             decoder_cell_state)
            # Store the output in tensorarray
            all_outputs = all_outputs.write(timestep, output)
        # Return the tensor array
        all_outputs = tf.transpose(all_outputs.stack(), (1, 0, 2))
        return all_outputs

#lossfunction
def custom_lossfunction(targets,logits):
    
    # Custom loss function that will not consider the loss for padded zeros.
    # Refer https://www.tensorflow.org/tutorials/text/nmt_with_attention#define_the_optimizer_and_the_loss_function
    
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    
    mask = tf.math.logical_not(tf.math.equal(targets, 0))
    
    loss_ = loss_object(targets, logits)
    
    mask = tf.cast(mask, dtype=loss_.dtype)
    
    loss_ *= mask
    
    return tf.reduce_mean(loss_)
        
#predict Encoder decoder
class pred_Encoder_decoder(tf.keras.Model): 
    def __init__(self, inp_vocab_size, out_vocab_size, embedding_dim, enc_units, dec_units, max_ita, max_eng, score_fun, att_units):
        #Intialize objects from encoder decoder
        super(pred_Encoder_decoder, self).__init__()
        self.encoder = Encoder(inp_vocab_size, embedding_dim, enc_units, max_ita)
        self.one_step_decoder = One_Step_Decoder(out_vocab_size, embedding_dim, max_eng, dec_units ,score_fun ,att_units)
        self.batch_size = batch_size
    def call(self, params):
        enc_inp = params[0]
        initial_state = self.encoder.initialize_states(1)
        output_state, enc_h, enc_c = self.encoder(enc_inp, initial_state)
        pred = tf.expand_dims([tokenizer_dec.word_index['<sos>']], 0)
        dec_h = enc_h
        dec_c = enc_c
        all_pred = []
        all_attention = []
        for t in range(50):  
            pred, dec_h,dec_c, attention, _ = self.one_step_decoder(pred, output_state, dec_h, dec_c)
            pred = tf.argmax(pred, axis = -1)
            all_pred.append(pred)
            pred = tf.expand_dims(pred, 0)
            all_attention.append(attention)
        return all_pred, all_attention


pred_model = pred_Encoder_decoder(vocab_size_enc, 
                                  vocab_size_dec, 
                                  embedding_dim, 
                                  lstm_size,
                                  lstm_size,
                                  max_enc, 
                                  max_dec, 
                                  'concat',
                                  att_units)

pred_model.compile(optimizer = 'Adam', loss = custom_lossfunction)

#Load the previously trained model
pred_model.load_weights('concat')


###################################################


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    seq = '<sos> '+ 'text' +' <eos>'
    seq = tokenizer_enc.texts_to_sequences([seq])
    seq = pad_sequences(seq, maxlen=max_enc, padding='post', dtype = np.int32)
    pred, attention_weights = pred_model.predict(tf.expand_dims(seq, 0))
    output = []
    for i in pred:
        word = tokenizer_dec.index_word[i[0]]
        if word == '<eos>':
            break
        output.append(word)

    return jsonify({'predicted text': ' '.join(output)})
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
