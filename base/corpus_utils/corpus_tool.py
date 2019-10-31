# -*- coding:utf-8 -*-
import codecs
import os
import data_config as dc

file_name = '../data/alldat.txt'

# for file in os.listdir(file_dir):

alldata_list = []
# print os.getcwd()+'/corpus/'+str(1)+'_.data'
for item in dc.Data_tup_scene:
    data_list = []
    new_data_list = []
    data_list=codecs.open(os.getcwd()+'/corpus/'+str(item[1])+'_.data')
    for line in data_list:
        line = str(item[1])+'\t'+line
        new_data_list.append(line)
    alldata_list.extend(new_data_list)
# alldata_list = list(set(alldata_list))
print len(alldata_list)
print alldata_list[9923]

with codecs.open(file_name,'w', 'utf-8') as w_file:
    for line in alldata_list:
        w_file.writelines(line)

print 'alldat.txt has already writen!'

