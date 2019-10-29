#!/usr/bin/env bash

#!/usr/bin/env bash
#export BERT_BASE_DIR=/home/zhaoxj/pycharmProjects/BERT-chinese-text-classification-and-deployment/bert/chinese_L-12_H-768_A-12
#export GLUE_DIR=/home/zhaoxj/pycharmProjects/BERT-chinese-text-classification-and-deployment/data
#export OUTPUT_DIR=/home/zhaoxj/pycharmProjects/BERT-chinese-text-classification-and-deployment/output
export BERT_BASE_DIR=/home/zxj/workspace/BERT-chinese-text-classification-and-deployment/bert/chinese_L-12_H-768_A-12
export GLUE_DIR=/home/zxj/workspace/BERT-chinese-text-classification-and-deployment/data
export OUTPUT_DIR=/home/zxj/workspace/BERT-chinese-text-classification-and-deployment/output

python classifier.py \
  --task_name=scene \
  --do_eval=true \
  --data_dir=$GLUE_DIR/ \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=32 \
  --test_batch_size=256 \
  --output_dir=$OUTPUT_DIR/