# -*- coding:utf-8 -*-
import data_config as dc

Data_tup_scene = dc.Data_tup_scene

label_list = []
count = 0
for item in Data_tup_scene:
    label_list.append(item[2])
    count+=1
print label_list
print count


