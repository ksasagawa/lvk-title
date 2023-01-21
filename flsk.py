from flask import Flask, render_template, request
import pandas as pd
import string
import numpy as np
import json
import tensorflow as tf
import keras.utils as ku
from keras.models import load_model
from random import randint

app = Flask('title-generator')

from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy

with open('cleaned.txt','r') as f:
    text = f.read().split("\n")

with open('card_archtypes.txt', 'r') as f:
    text = text + (f.read().split("\n"))

def clean_text(txt):
    txt = "".join(v for v in txt if v not in string.punctuation).lower()
    txt = txt.encode("utf8").decode("ascii",'ignore')
    return txt 

corpus = [clean_text(x) for x in text]

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
inp_sequences, total_words = get_sequence_of_tokens(corpus)
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

def generate_padded_sequences(input_sequences):
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, num_classes=total_words)
    return predictors, label, max_sequence_len

predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences)

@app.route('/')
def show_predict_stock_form():
    return render_template('holder.html')
@app.route('/result', methods=['POST'])
def results():
    form = request.form
    if request.method == 'POST':
       modelName = 'model' + str(randint(1,5))
       model = load_model(modelName)
       prompt = request.form["prompt"]
       result = generate_text(prompt, 7, model, max_sequence_len)
       return render_template('holder2.html', result=result)

app.run()