#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-07-23 13:25:05
# @Author  : Lee (lijingyang@tcl.com)

import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os.path
from random import shuffle
import data_config as dc
import time


def write_data_to_txt(data, file_dir):
    """
    :param file_dir: 
    :return: 
    """
    try:
        import os
        dirs, file_name = os.path.split(file_dir)
        if not os.path.isdir(dirs):
            os.makedirs(dirs)
        fp = open(file_dir, "w")
        for item in data:
            item = str(item)
            item = item.replace(' ', '')
            fp.write(item + "\n")
        fp.close()
        print "write %d into %s" % (len(data), file_dir)
    except IOError:
        print("fail to open file")


def read_data_from_txt(file_dir):
    """
    :param file_dir: 
    :return: 
    """
    data_list = []
    lines = open(file_dir, 'r').readlines()
    for line in lines:
        line = line.decode('utf-8').split('\n')
        if '' in line:
            line.remove('')
        line = ''.join(line).replace(" ", "")
        if '' != line:
            data_list.append(line)
    return data_list


def search_file(dir, filename):
    for parent, dirnames, filenames in os.walk(dir):
        for file in filenames:
            if file == filename:
                return os.path.join(parent, file)


def extend_data(data, target_num):
    data_len = len(data)
    datas = []
    repeat = target_num / data_len
    for i in range(repeat):
        datas.extend(data)
    return datas


def make_corpus(input_dir, output_dir, data_tup, input_dir_2=None, max_ratio=3):
    """
    Function: make_corpus
    Summary: make datas for model
    Attributes: 
        @param (input_dir): corpus_floder
        @param (output_dir): output dir
        @param (data_tup):InsertHere
        @param (input_dir_2) default=None: InsertHere
        @param (max_ratio): len(generate_datas)/ len(real_datas)
    Returns: None, corpus in output floder
    """
    for item in data_tup:
        # 安领域循环，一个领域有若干个文件
        # item: (['map_scenic.data'], 2, 'map_scenic') 是tuple
        # item[0]:['map_scenic.data'] 是一个list
        data_list = item[0]
        datas = []
        for data_name in data_list:
            data_path = search_file(input_dir, data_name)
            if data_path != None:
                # data 是一个list 长度为 该文件所含语料的条数
                data = read_data_from_txt(data_path)

            else:
                data = []
                print "Not found file: " + data_name + " in dir: " + input_dir

            # make real_corpus mixed up with generated corpus
            if input_dir_2 != None:
                data_path_2 = search_file(input_dir_2, data_name)
                if None != data_path_2:

                    data2 = read_data_from_txt(data_path_2)
                    ratio = len(data) / len(data2)
                    if ratio > max_ratio:
                        print "repeat real data %d times !" % ratio
                        for i in range(ratio):
                            data.extend(data2)
                    else:
                        data.extend(data2)

            datas.extend(data)

        write_data_to_txt(datas, output_dir+str(item[1])+'_.data')


def mixed_corpus(input_dir_1, input_dir_2, max_ratio_tup, out_dir):
    # input_dir_1 should be the original data path
    for item in max_ratio_tup:
        print "build data: " + str(item[1]) + ', '+dc.Data_tup_v4[item[1]][2]
        datas = []
        file_name = str(item[1]) + '_.data'
        data_path_1 = search_file(input_dir_1, file_name)
        data_1 = read_data_from_txt(data_path_1)
        data_path_2 = search_file(input_dir_2, file_name)
        data_2 = read_data_from_txt(data_path_2)
        if len(data_1) * len(data_2) != 0:
            mix_ratio = item[0]
            print "max_ratio:" + str(mix_ratio)
            datas.extend(data_2)
            data_2_length = len(data_2)
            print "data_2 length::%d"%data_2_length
            if mix_ratio ==1:
                pass
            else:
                if mix_ratio != 0:
                    data_1_length = int(data_2_length / mix_ratio * (1 - mix_ratio))
                    shuffle(data_1)
                    print "data_1 length:: %d"%len(data_1)
                    print "use data_1 length::%d"%data_1_length
                    datas.extend(data_1[:data_1_length])
                else:
                    datas.extend(data_1)
            shuffle(datas)
        else:
            print "file is null in dir_1 or dir_2"
            datas.extend(data_1)
            datas.extend(data_2)
            shuffle(datas)
        write_data_to_txt(datas, out_dir+str(item[1]) + '_.data')


def split_data(input_dir, min_sample_num, ratio=0.1):

    list_filename = []
    for parent, dirnames, filenames in os.walk(input_dir):
        index = 0
        for filename in filenames:
            list_filename.append((filename, index))
            index += 1
            file_path = os.path.join(parent, filename)
            data = read_data_from_txt(file_path)
            data_len = len(data)
            if data_len > 100:
                shuffle(data)
                split_num = int(round(data_len * (1 - ratio)))
            if data_len < (1/ratio):
                split_num = data_len - 2
            train_data = []
            test_data = []
            for i in range(split_num):
                train_data.append(data[i])
            for i in range(split_num, data_len):
                test_data.append(data[i])

            if len(train_data) < min_sample_num:
                train_data = extend_data(train_data, min_sample_num)
                #test_data = extend_data(test_data, int(min_sample_num*ratio))

            write_data_to_txt(train_data, './train_data/' +
                              'train_' + filename)
            write_data_to_txt(test_data, './test_data/' + 'test_' + filename)


def make_eval_datas_and_labels(dir, data_tup, out_dir):
    datas = []
    labels = []

    if dir.find('test') != -1:
        file_name = 'test_'
    elif dir.find('train') != -1:
        file_name = 'train_'
    else:
        print "unkown dir !"
        return

    for item in data_tup:
        data = read_data_from_txt(dir + file_name+str(item[1])+'_.data')
        data = list(set(data))
        label = [item[1] for i in range(len(data))]
        datas.extend(data)
        labels.extend(label)
    write_data_to_txt(datas, out_dir + file_name + 'eval.datas')
    write_data_to_txt(labels, out_dir + file_name + 'eval.labels')


def extend_corpus(files_dir, item_length):
    """
    Function: repeat_item, in oder to raise short sentences porportion in data 
    """
    print "extend_corpus"
    for parent, dirnames, filenames in os.walk(files_dir):
        for filename in filenames:
            if filename != 'train_22_.data' and filename != 'train_23_.data' \
                    and filename != 'train_2_.data' and filename != 'train_0_.data':
                file_path = os.path.join(parent, filename)
                datas = read_data_from_txt(file_path)
                extend_datas = []
                for sen in datas:
                    if len(sen) <= item_length:
                        extend_sen = sen + sen
                        # print extend_sen
                        extend_datas.append(sen)
                        extend_datas.append(extend_sen)

                extend_datas.extend(datas)
                shuffle(extend_datas)
                write_data_to_txt(extend_datas, file_path)


if __name__ == '__main__':

    #生成data/corpus 目录下的语料
    make_corpus(dc.Original_corpus_dir, dc.Corpus_dir, dc.Data_tup_scene)

    #make_corpus(dc.Original_corpus_dir, './corpus_1/', dc.Data_tup_v4)
    #make_corpus(dc.Preprocess_corpus_dir, './corpus_2/', dc.Data_tup_v5)
    #mixed_corpus('./corpus_1/', './corpus_2/',
    #             dc.Mixed_ratio_define, dc.Corpus_dir)
    
    # 把语料切分成 train_corpus 和 test_corpus
    split_data(dc.Corpus_dir, 40000, ratio=0.0001)
    
    make_eval_datas_and_labels("./test_data/", dc.Data_tup_scene, "./test_data/")
    make_eval_datas_and_labels("./train_data/", dc.Data_tup_scene, "./test_data/")
    extend_corpus('./train_data/', 4)

    #search_file(corpus_dir, 'play_control.data')
