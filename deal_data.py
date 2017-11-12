import jieba
import numpy as np
import nltk
import itertools
import random
import sys
from collections import defaultdict
import pickle
import re
from zhon.hanzi import punctuation
UNK = 'unk'
data_path = 'd:/directory_sub/new_faqs_data_v2.txt'
limit = {
        'maxq' : 30,
        'minq' : 0,
        'maxa' : 100,
        'mina' : 3
        }
def throw_dirty(sentence):
    content=re.sub("[%s]+" %punctuation, "", sentence)
    newline2 = re.sub("[A-Za-z0-9\[\`\~\!\@\#\$\^\&\*\(\)\=\|\{\}\'\:\;\'\,\[\]\.\<\>\/\?\~\！\@\#\\\&\*\%\-\_]", "", content)
    newline3 = re.sub(' ','',newline2)
    return newline3

def get_data():
    w = open(data_path, 'r', encoding='utf-8')
    all_ques = []
    all_ans = []
    all_sen = []
    for line in w.readlines():
        newline = line.strip().split('----')
        newline =[throw_dirty(i) for i in newline]
        if len(list(jieba.cut(newline[2]))) > 30 or len(list(jieba.cut(newline[3]))) > 100:
            continue
        all_ques.append(list(jieba.cut(newline[2])))
        all_ans.append(list(jieba.cut(newline[3])))
        all_sen.append(list(jieba.cut(newline[2]))+list(jieba.cut(newline[3])))
    w.close()
    return all_ques , all_ans , all_sen


def index_(tokenized_sentences, vocab_size):
    # get frequency distribution
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    # get vocabulary of 'vocab_size' most used words
    vocab = freq_dist.most_common(vocab_size)
    # index2word
    index2word = ['_'] + [UNK] + [ x[0] for x in vocab ]
    # word2index
    word2index = dict([(w,i) for i,w in enumerate(index2word)] )
    return index2word, word2index, freq_dist


def zero_pad(qtokenized, atokenized, w2idx):
    # num of rows
    data_len = len(qtokenized)
    # numpy arrays to store indices
    idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32)
    idx_a = np.zeros([data_len, limit['maxa']], dtype=np.int32)
    for i in range(data_len):
        q_indices = pad_seq(qtokenized[i], w2idx, limit['maxq'])
        a_indices = pad_seq(atokenized[i], w2idx, limit['maxa'])
        #print(len(idx_q[i]), len(q_indices))
        #print(len(idx_a[i]), len(a_indices))
        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)
    return idx_q, idx_a



def pad_seq(seq, lookup, maxlen):
    indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])
    return indices + [0]*(maxlen - len(seq))


def process_data():
    all_ques, all_ans, all_sen =  get_data()
    # # convert list of [lines of text] into list of [list of words ]
    # print('\n>> Segment lines into words')
    # print('\n:: Sample from segmented list of words')
    # print('\nq : {0} ; a : {1}'.format(qtokenized[60], atokenized[60]))
    # print('\nq : {0} ; a : {1}'.format(qtokenized[61], atokenized[61]))
    # indexing -> idx2w, w2idx : en/ta
    print('\n >> Index words')
    idx2w, w2idx, freq_dist = index_( all_sen, vocab_size=6000)
    print('\n >> Zero Padding')
    idx_q, idx_a = zero_pad(all_ques, all_ans, w2idx)
    print(idx_q[0])
    print('\n >> Save numpy arrays to disk')
    # save them
    np.save('idx_q.npy', idx_q)
    np.save('idx_a.npy', idx_a)
    # let us now save the necessary dictionaries
    metadata = {
            'w2idx' : w2idx,
            'idx2w' : idx2w,
            'limit' : limit,
            'freq_dist' : freq_dist
                }
    # write to disk : data control dictionaries
    with open('metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)


def load_data(PATH=''):
    # read data control dictionaries
    with open(PATH + 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    # read numpy arrays
    idx_q = np.load(PATH + 'idx_q.npy')
    idx_a = np.load(PATH + 'idx_a.npy')
    return metadata, idx_q, idx_a
if __name__ == '__main__':
    process_data()

