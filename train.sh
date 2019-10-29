#!/usr/bin/env bash
#export BERT_BASE_DIR=/home/zhaoxj/pycharmProjects/albert_zh/albert_base_zh
#export GLUE_DIR=/home/home/zhaoxj/pycharmProjects/albert_zh/data
#export OUTPUT_DIR=/home/zhaoxj/pycharmProjects/albert_zh/output
export BERT_BASE_DIR=/home/zxj/workspace/albert_zh/albert_base_zh

export GLUE_DIR=/home/zxj/workspace/albert_zh/data
export OUTPUT_DIR=/home/zxj/workspace/albert_zh/output


python run_classifier.py \
  --task_name=scene \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/ \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/albert_config_base.json \
  --init_checkpoint=$BERT_BASE_DIR/albert_model.ckpt \
  --max_seq_length=32 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=4.0 \
  --output_dir=$OUTPUT_DIR/
