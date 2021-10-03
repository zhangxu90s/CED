import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)

import sys

if not '../' in sys.path: sys.path.append('../')

import pandas as pd

from utils import data_utils
from model_config import config
from mmi import MmiSeq2SeqModel

from sklearn.model_selection import train_test_split
import numpy as np
def train_model(config):
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

    print('[INFO] Preparing data for experiment: {}'.format(config['experiment']))
    train_data, _x_val_test = train_test_split(x, test_size = 0.1, random_state = 10)
    val_data, test_data = train_test_split(_x_val_test, test_size = 0.5, random_state = 10)

    true_val = val_data
    filters = '!"#$%&()*+/:;<=>@[\\]^`{|}~\t\n'
    w2v_path = config['w2v_dir'] + 'w2v_300d_snli_all_sentences.pkl'

    print('[INFO] Split data into train-validation-test sets')
    x_train, x_val, x_test = train_data, val_data, test_data 

    encoder_embeddings_matrix = data_utils.create_embedding_matrix(word_index,
                                                                   config['embedding_size'],
                                                                   w2v_path)

    decoder_embeddings_matrix = data_utils.create_embedding_matrix(word_index,
                                                                   config['embedding_size'],
                                                                   w2v_path)

    # Re-calculate the vocab size based on the word_idx dictionary
    config['encoder_vocab'] = len(word_index)
    config['decoder_vocab'] = len(word_index)

    model = MmiSeq2SeqModel(config,
                                   encoder_embeddings_matrix,
                                   decoder_embeddings_matrix,
                                   word_index,
                                   word_index)

    model.train(x_train, x_val, true_val)


if __name__ == '__main__':
    train_model(config)