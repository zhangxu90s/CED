# -*- coding: utf-8 -*-

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

import sys

if not '../' in sys.path: sys.path.append('../')

from utils import data_utils
from model_config import config
from mmi import MmiSeq2SeqModel
from sklearn.model_selection import train_test_split
import numpy as np

snli_data = data_utils.get_sentences(file_path = config['data'])

print('[INFO] Number of sentences = {}'.format(len(snli_data)))

sentences = [s.strip() for s in snli_data]

np.random.shuffle(sentences)

print('[INFO] Tokenizing input and output sequences')
filters = '!"#$%&()*+/:;<=>@[\\]^`{|}~\t\n'
x, word_index = data_utils.tokenize_sequence(sentences,
                                             filters,
                                             config['num_tokens'],
                                             config['vocab_size'])

train_data, _x_val_test = train_test_split(x, test_size = 0.1, random_state = 10)
val_data, test_data = train_test_split(_x_val_test, test_size = 0.5, random_state = 10)

true_test = test_data
input_test = test_data
filters = '!"#$%&()*+/:;<=>@[\\]^`{|}~\t\n'
w2v_path = config['w2v_dir'] + 'w2v_300d_snli_all_sentences.pkl'

x_train, x_val, x_test = train_data, val_data, test_data 

encoder_embeddings_matrix = data_utils.create_embedding_matrix(word_index, config['embedding_size'], w2v_path)

decoder_embeddings_matrix = data_utils.create_embedding_matrix(word_index, config['embedding_size'], w2v_path)

# Re-calculate the vocab size based on the word_idx dictionary
config['encoder_vocab'] = len(word_index)
config['decoder_vocab'] = len(word_index)

model = MmiSeq2SeqModel(config, 
                               encoder_embeddings_matrix, 
                               decoder_embeddings_matrix, 
                               word_index, 
                               word_index)
if config['load_checkpoint'] != 0: 
    checkpoint = config['model_checkpoint_dir'] + str(config['load_checkpoint']) + '.ckpt'
else:
    checkpoint = tf.train.get_checkpoint_state(os.path.dirname('models/checkpoint')).model_checkpoint_path

preds = model.predict(checkpoint, x_test, true_test)
count = 100
model.show_output_sentences(preds[:count], 
                            input_test[:count], 
                            true_test[:count])
model.get_diversity_metrics(checkpoint, x_test)
