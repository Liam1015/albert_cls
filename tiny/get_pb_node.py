# -*- coding:utf-8 -*-
import tensorflow as tf
import os

model_dir = './model_pb'
model_name = 'cls_scene.pb'
# model_dir = './model_pb/1572250528'
# model_name = 'saved_model.pb'


def create_graph():
    with tf.gfile.GFile(os.path.join(model_dir, model_name), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

create_graph()
tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
print len(tensor_name_list)
for tensor_name in tensor_name_list:
    print(tensor_name)

