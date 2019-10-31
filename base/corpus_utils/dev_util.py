# -*- coding:utf-8 -*-
import codecs
import os
import data_config as dc

file_name = '../data/dev.tsv'

# for file in os.listdir(file_dir):

alldata_list = []
# print os.getcwd()+'/corpus/'+str(1)+'_.data'
for item in dc.Data_tup_scene:
    data_list = []
    new_data_list = []
    data_list=codecs.open(os.getcwd()+'/test_data/'+'test_'+str(item[1])+'_.data')
    for line in data_list:
        line = str(item[2]).strip()+'\t'+line.strip()
        new_data_list.append(line)
    alldata_list.extend(new_data_list)
# alldata_list = list(set(alldata_list))
print len(alldata_list)


with codecs.open(file_name,'w', 'utf-8') as w_file:
    for line in alldata_list:
        w_file.writelines(line+'\n')

print 'dev.tsv has already writen!'

