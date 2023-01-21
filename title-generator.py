import pandas as pd
import string
import numpy as np
import json
import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku
tf.random.set_seed(2)
from numpy.random import seed
seed(1)


with open('cleaned.txt','r') as f:
    text = f.read().split("\n")

with open('card_archtypes.txt', 'r') as f:
    text = text + (f.read().split("\n"))

def clean_text(txt):
    txt = "".join(v for v in txt if v not in string.punctuation).lower()
    txt = txt.encode("utf8").decode("ascii",'ignore')
    return txt 

corpus = [clean_text(x) for x in text]
print(corpus[:10])


tokenizer = Tokenizer()
def get_sequence_of_tokens(corpus):
    #get tokens
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
 
#convert to sequence of tokens
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
 
    return input_sequences, total_words
inp_sequences, total_words = get_sequence_of_tokens(corpus)



def generate_padded_sequences(input_sequences):
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, num_classes=total_words)
    return predictors, label, max_sequence_len

predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences)



def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()
    
    # Add Input Embedding Layer
    model.add(Embedding(total_words, 10, input_length=input_len))
    
    # Add Hidden Layer 1 - LSTM Layer
    model.add(LSTM(100))
    model.add(Dropout(0.1))
    
    # Add Output Layer
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model


seed(1)
model1 = create_model(max_sequence_len, total_words)
seed(2)
model2 = create_model(max_sequence_len, total_words)
seed(3)
model3 = create_model(max_sequence_len, total_words)
seed(4)
model4 = create_model(max_sequence_len, total_words)
seed(5)
model5 = create_model(max_sequence_len, total_words)
model1.summary()

models = [model1,model2,model3,model4,model5]

model1.fit(predictors, label, epochs=100, verbose=5)
model2.fit(predictors, label, epochs=100, verbose=5)
model3.fit(predictors, label, epochs=100, verbose=5)
model4.fit(predictors, label, epochs=100, verbose=5)
model5.fit(predictors, label, epochs=100, verbose=5)

i = 1
for model in models:
    model.save('model' + str(i))
    i += 1

import numpy
def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose='2')
        output_word = ""
        for word,index in tokenizer.word_index.items():
            if index == numpy.argmax(predicted):
                output_word = word
                break
        seed_text += " "+output_word
    return seed_text.title()
