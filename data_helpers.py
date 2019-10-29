import numpy as np
import codecs
import re
import os
from collections import defaultdict
import pickle
import data.data_config as dc
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from random import shuffle
import random
import operator
from tqdm import tqdm


def build_vocabulary(sentences, savedir):
    print "begin to build vocab"

    pad_str = '[PAD]'.decode('utf-8')
    unk_str = '[UNK]'.decode('utf-8')

    # just for count_words
    all_words = []
    for sen in sentences:
        for word in sen:
            if " " != word:
                all_words.append(word)
    print("all_words.len::", len(all_words))
    # sort wordlist with word's count
    word_set = list(set(all_words))
    print("word_set.len::", len(word_set))

    count_set_list = {}
    for word in all_words:
        if count_set_list.has_key(word):
            count_set_list[word] += 1
        else:
            count_set_list[word] = 1
        #count_set_list[item] = all_words.count(item)

    sorted_list_word = sorted(count_set_list.items(),
                              key=operator.itemgetter(1), reverse=False)

    word_list = []
    for item in sorted_list_word[::-1]:
        word_list.append(item[0])

    word_list.append(unk_str)
    if pad_str in word_list:
        word_list.remove(pad_str)
    word_list.insert(0, pad_str)

    word_dic = dict()

    with open(savedir, 'w') as f:
        for word in word_list:
            word_dic[word] = int(word_list.index(word))
            line = str(word) + "\n"
            f.write(line)

    return word_dic, len(word_dic.keys())


def read_vocabulary(voc_dir):

    voc = dict()
    lines = open(voc_dir, 'r').readlines()

    for i in range(len(lines)):
        key = lines[i].decode('utf-8').split('\n')[0]
        voc[key] = i

    print 'read vocabulary len : %f' % len(voc.keys())
    return voc, len(voc.keys())


def sentence2matrix(sentences, max_length, vocs):
    sentences_num = len(sentences)
    data_dict = np.zeros((sentences_num, max_length), dtype='int32')

    for index, sentence in enumerate(sentences):
        data_dict[index, :] = map2id(sentence, vocs, max_length)

    return data_dict


def map2id(sentence, voc, max_len):
    array_int = np.zeros((max_len,), dtype='int32')
    min_range = min(max_len, len(sentence))

    for i in range(min_range):
        item = sentence[i]
        array_int[i] = voc.get(item, voc['[UNK]'])

    return array_int

def id2sentence(x_list, voc):
    revese_voc = {v: k for k, v in voc.items()}
    word_list = []
    for id in x_list:
        word = revese_voc.get(id, revese_voc[len(revese_voc)-1])
        word_list.append(word)
        sentence = ''.join([word for word in word_list])
    return sentence

def get_sample_label(y_list, Data_tup_scene):
    index = y_list.index(1)
    label = Data_tup_scene[index][2]
    return label

def get_pred_lable(pred, Data_tup_scene):
    index = pred[0]
    label = Data_tup_scene[index][2]
    return label

def clean_str(string):
    return string.strip().lower()


def mkdir_if_not_exist(dirpath):

    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    return dirpath


def seperate_line(line):
    return ''.join([word + ' ' for word in line])


def read_and_clean_file(input_file, output_cleaned_file=None):
    lines = list(open(input_file, "r").readlines())
    lines = [clean_str(seperate_line(line.decode('utf-8'))) for line in lines]

    if output_cleaned_file is not None:
        with open(output_cleaned_file, 'w') as f:
            for line in lines:
                f.write((line + '\n').encode('utf-8'))

    return lines


def load_data_and_labels(data_tup, data_dir):
    datas = []
    labels = []
    for item in data_tup:
        data = read_and_clean_file(data_dir+'train_'+ str(item[1])+ '_.data')
        datas += data
        label = [[0 for i in range(len(data_tup))] for _ in data]
        for la in label:
            la[item[1]] = 1
        labels += label
    labels = np.array(labels)

    return [datas, labels]


def load_testfile_and_labels(input_text_file, input_label_file, num_samples=-1):
    x_text = read_and_clean_file(input_text_file)

    y = None if not os.path.exists(input_label_file) else map(
        int, list(open(input_label_file, "r").readlines()))

    # get some samples randomly form testfile, -1 means all samples
    if num_samples != -1:
        index_list = [i for i in range(len(x_text))]
        samples_index = random.sample(index_list, num_samples)
        samples = []
        labels = []
        for index in samples_index:
            samples.append(x_text[index])
            labels.append(y[index])
        x_text = samples
        y = labels

    return (x_text, y)


def padding_sentences(input_sentences, padding_token, padding_sentence_length=None):
    sentences = [sentence.split(' ') for sentence in input_sentences]
    max_sentence_length = padding_sentence_length if padding_sentence_length is not None else max(
        [len(sentence) for sentence in sentences])

    for sentence in sentences:
        if len(sentence) > max_sentence_length:
            sentence = sentence[:max_sentence_length]
        else:
            sentence.extend([padding_token] *
                            (max_sentence_length - len(sentence)))

    return sentences


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            yield shuffled_data[start_index:end_index]


def read_data_from_strs(lines, max_sentence_length):
    data_line = []

    for line in lines:
        line = line.decode('utf-8')
        line = ''.join([word + ' ' for word in line])
        line = line.strip().lower()
        line = line.split(' ')

        if len(line) > max_sentence_length:
            line = line[:max_sentence_length]
        else:
            line.extend(['<pad>'] * (max_sentence_length - len(line)))

        data_line.append(line)

    return data_line


def read_data_from_str(line, max_sentence_length):
    line = line.decode('utf-8')
    line = ''.join([word + ' ' for word in line])
    line = line.strip().lower()
    line = line.split(' ')

    if len(line) > max_sentence_length:
        line = line[:max_sentence_length]
    else:
        line.extend(['<pad>'] * (max_sentence_length - len(line)))

    return [line]


def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See 5.4 and https://arxiv.org/abs/1512.00567.
    inputs: 3d tensor. [N, T, V], where V is the number of vocabulary.
    epsilon: Smoothing rate.

    For example,

    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1],
       [0, 1, 0],
       [1, 0, 0]],

      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)

    outputs = label_smoothing(inputs)

    with tf.Session() as sess:
        print(sess.run([outputs]))

    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],

       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]
    ```
    '''
    V = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / V)



if __name__ == '__main__':
    pass
