######################################################################################################################################################################
########################################################### Import Packages ##########################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Embedding,Dropout,GRU
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

######################################################################################################################################################################
############################################################ Data Preprocessing ######################################################################################

text = open('TrumpSpeak/trumpspeak.txt', 'r').read()
vocab = sorted(set(text))
char_to_ind = {u:i for i, u in enumerate(vocab)}
ind_to_char = np.array(vocab)
encoded_text = np.array([char_to_ind[c] for c in text])
seq_len = 120
total_num_seq = len(text)//(seq_len+1)
char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)
sequences = char_dataset.batch(seq_len+1, drop_remainder=True)

def create_seq_targets(seq):
  input_txt = seq[:-1]
  target_txt = seq[1:]
  return(input_txt, target_txt)

dataset = sequences.map(create_seq_targets)
batch_size = 128
buffer_size = 10000
dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
vocab_size = len(vocab)
embed_dim = 64
rnn_neurons = 1026
early_stopping = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=3)

######################################################################################################################################################################
############################################################## Model Construction ####################################################################################

def sparse_cat_loss(y_true,y_pred):
  return sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

def create_model(vocab_size, embed_dim, rnn_neurons, batch_size):
  model = Sequential()
  model.add(Embedding(vocab_size, embed_dim,batch_input_shape=[batch_size, None]))
  model.add(GRU(rnn_neurons,return_sequences=True,stateful=True,recurrent_initializer='glorot_uniform',reset_after=False))
  model.add(Dense(vocab_size))
  model.compile(optimizer='adam', loss=sparse_cat_loss) 
  return model

model = create_model(
  vocab_size = vocab_size,
  embed_dim=embed_dim,
  rnn_neurons=rnn_neurons,
  batch_size=batch_size)
model.summary()

######################################################################################################################################################################
############################################################### Model Fitting/Saving #################################################################################

epochs = 100
model.fit(dataset,epochs=epochs,callbacks=[early_stopping])
model.save('TrumpSpeak/trumptalk.h5') 

######################################################################################################################################################################
############################################################### Model Visualization ##################################################################################

plt.plot(model.history.history['loss'])
plt.show()

######################################################################################################################################################################
################################################################ Make new model with old weights #####################################################################

model = create_model(vocab_size, embed_dim, rnn_neurons, batch_size=1)
model.load_weights('TrumpSpeak/trumptalk.h5')
model.build(tf.TensorShape([1, None]))
model.summary()

######################################################################################################################################################################
################################################################ Generate Text #######################################################################################

def generate_text(model, start_seed, gen_size=100, temp=1.0):
  num_generate = gen_size
  input_eval = [char_to_ind[s] for s in start_seed]
  input_eval = tf.expand_dims(input_eval, 0)
  text_generated = []
  temperature = temp
  model.reset_states()
  for i in range(num_generate):
    predictions=model(input_eval)
    i = i
    predictions=tf.squeeze(predictions,0)
    predictions=predictions/temperature
    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
    input_eval=tf.expand_dims([predicted_id],0)
    text_generated.append(ind_to_char[predicted_id])
  return(start_seed + ''.join(text_generated))

print(generate_text(model,'Trump',gen_size=1000))