# coding:utf-8
# Restore model from pb file and do prediction

import sys
import codecs
import numpy as np
import tensorflow as tf
from tensorflow.contrib import predictor
import tokenization
import time

from pathlib import Path

PROJECT_PATH = Path(__file__).absolute().parent
sys.path.insert(0, str(PROJECT_PATH))


flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("vocab_file", "albert_base_zh/vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")


class bertPredict(object):
    def __init__(self, pb_path, vocab_path):
        subdirs = [x for x in Path(pb_path).iterdir()
                   if x.is_dir() and 'temp' not in str(x)]
        latest = str(sorted(subdirs)[-1])

        self.vocab_idx, self.idx_vocab = self._load_vocab(vocab_path)
        self.predict_fn = predictor.from_saved_model(latest)


    def predict(self, inputs, max_seq_length):

        input_ids, input_mask, segment_ids, label_ids = self._process_input(inputs, max_seq_length)

        input_ids = np.array(input_ids, dtype=np.int32)
        input_mask = np.array(input_mask, dtype=np.int32)
        segment_ids = np.array(segment_ids, dtype=np.int32)
        label_ids = np.array(label_ids, dtype=np.int32)
        print input_ids
        print input_mask
        print segment_ids
        print label_ids
        start = time.time()
        result = self.predict_fn(
            {'input_ids': input_ids,
             'input_mask': input_mask,
             'segment_ids': segment_ids,
             'label_ids': label_ids}
        )
        end = time.time()
        print "cost time is %f" % (end - start)
        return result

    def _process_input(self, inputs, max_seq_length):
        tokenizer = tokenization.FullTokenizer(
            vocab_file=FLAGS.vocab_file, do_lower_case=True)
        tokens_text = tokenizer.tokenize(inputs)
        if len(tokens_text) > max_seq_length - 2:
            tokens_text = tokens_text[0:(max_seq_length - 2)]
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_text:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        # label_list =
        # label_map = {}
        # for (i, label) in enumerate(label_list):
        #     label_map[label] = i
        label_ids = 0

        return [input_ids], [input_mask], [segment_ids], [label_ids]

    def _load_vocab(self, vocab_path):
        with codecs.open(vocab_path, 'r', 'utf-8') as file:
            vocab_idx = {}
            idx_vocab = {}
            for idx, vocab in enumerate(file):
                vocab = vocab.strip()
                idx = int(idx)
                vocab_idx[vocab] = idx
                idx_vocab[idx] = vocab
        return vocab_idx, idx_vocab


if __name__ == '__main__':
    label_list = ['control', 'app_control', 'video', 'weather', 'channel_switch',
                'image_interactive', 'math_operation', 'disport', 'time_query', 'baike',
                'info_news', 'info_stock&fund', 'translate', 'converter', 'karaoke',
                'words', 'audio', 'map_server', 'ticket_server', 'order_server',
                'third_query', 'multi_dialogue', 'unknown', 'search', 'couplet', 'poetry',
                'relation_dialogue', 'album', 'help']

    bert = bertPredict('./model_pb', 'albert_base_zh/vocab.txt')

    result = bert.predict('下降', max_seq_length=128)


    result = result['output'][0]
    label_id = np.argmax(result)
    print label_id
    label = label_list[label_id]
    prob = result[label_id]
    print "label:"+ label +'\t'+"prob:%.2f"%prob

    # for idx in result['output']:
    #     print(bert.idx_vocab[idx])