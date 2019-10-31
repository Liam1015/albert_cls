# -*- coding:utf-8 -*-
import codecs
import os
import data_config as dc
import random

file_name = '../data/test.tsv'

# for file in os.listdir(file_dir):

alldata_list = []
# print os.getcwd()+'/corpus/'+str(1)+'_.data'
for item in dc.Data_tup_scene:
    new_data_list = []
    data_list = codecs.open(os.getcwd()+'/train_data/'+'train_'+str(item[1])+'_.data').readlines()
    for line in data_list:
        line = str(item[2]).strip()+'\t'+line.strip()
        new_data_list.append(line)
    alldata_list.extend(new_data_list)
# alldata_list = list(set(alldata_list))
print len(alldata_list)
print alldata_list[9923]
print alldata_list[9923]

with codecs.open(file_name,'w', 'utf-8') as w_file:
    for line in alldata_list:
        w_file.writelines(line+'\n')

print 'test.tsv has already writen!'

