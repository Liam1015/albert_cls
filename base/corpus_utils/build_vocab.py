#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-10-30 14:42:48
# @Author  : Lee (lijingyang@tcl.com)

import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import data_utils as du
import data_config as dc
import operator

def build_vocabulary(corpus_dir, savedir, remove_sen_dir=None):

    sentences = []
    for parent, dirnames, filenames in os.walk(corpus_dir):
        for filename in filenames:
            file_path = os.path.join(parent, filename) 
            _sens = du.read_data_from_txt(file_path)
            sentences.extend(_sens)
    if remove_sen_dir != None:
        remove_sens = du.read_data_from_txt(remove_sen_dir)
        sentences = list(set(sentences) - set(remove_sens))
    else:
        sentences = list(set(sentences))
        print("length of sentences is :", len(sentences))

    print "begin to build vocab"

    pad_str = '<pad>'.decode('utf-8')
    unk_str = '<unk>'.decode('utf-8')

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
        # print("type of item is ", type(item))
        word_list.append(item[0])

    # print "lenght of word_list is ", len(word_list)

    word_list.append(unk_str)
    if pad_str in word_list:
        word_list.remove(pad_str)
    word_list.insert(0, pad_str)

    word_dic = dict()

    if not os.path.exists(savedir):
        os.mkdir(savedir)

    with open(savedir+'/vocab', 'w') as f:
        for word in word_list:
            word_dic[word] = int(word_list.index(word))
            line = str(word) + "\n"
            f.write(line)


if __name__ == '__main__':
    #build_vocabulary('./corpus', '../runs', dc.Original_corpus_dir+'unknown/unknown_neg.data')
    build_vocabulary('./corpus', '../runs')