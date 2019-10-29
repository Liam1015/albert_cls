#!/usr/bin/env bash
#export BERT_BASE_DIR=/home/zhaoxj/pycharmProjects/BERT-chinese-text-classification-and-deployment/bert/chinese_L-12_H-768_A-12
#export GLUE_DIR=/home/zhaoxj/pycharmProjects/BERT-chinese-text-classification-and-deployment/data
#export MODEL_DIR=/home/zhaoxj/pycharmProjects/BERT-chinese-text-classification-and-deployment/output
#export MODEL_PB_DIR=/home/zhaoxj/pycharmProjects/BERT-chinese-text-classification-and-deployment/model_pb/

export BERT_BASE_DIR=/home/zhaoxj/pycharmProjects/albert_zh_base/albert_base_zh
export GLUE_DIR=/home/zhaoxj/pycharmProjects/albert_zh_base/data
export MODEL_DIR=/home/zhaoxj/pycharmProjects/albert_zh_base/output
export MODEL_PB_DIR=/home/zhaoxj/pycharmProjects/albert_zh_base/model_pb

python export.py \
  --task_name=scene \
  --do_predict=true \
  --data_dir=$GLUE_DIR/ \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/albert_config_base.json \
  --model_dir=$MODEL_DIR/ \
  --serving_model_save_path=$MODEL_PB_DIR



