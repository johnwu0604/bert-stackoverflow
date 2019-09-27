import tensorflow as tf
import numpy as np
import json
from tensorflow.python.saved_model import tag_constants
import pandas as pd
from bert import tokenization
import run_stackoverflow_classifier
import os

# TODO: Point these to remote locations
max_seq_length = 128
export_dir = '../scoring/export' # should use registered model
predict_file = 'predict.tf_records'
vocab_file = '../scoring/vocab.txt' # should use blob storage location
labels = ['c#', '.net', 'java', 'asp.net', 'c++', 'javascript', 'php', 'python', 'sql', 'sql-server'] # should use blob storage location

def init():
    global sess, tensor_input_ids, tensor_input_mask, tensor_label_ids, tensor_segment_ids, tensor_outputs, tokenizer
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
    tf.reset_default_graph()
    sess = tf.Session()
    tf.saved_model.loader.load(sess, [tag_constants.SERVING], export_dir)
    tensor_input_ids = tf.get_default_graph().get_tensor_by_name('input_ids_1:0')
    tensor_input_mask = tf.get_default_graph().get_tensor_by_name('input_mask_1:0')
    tensor_label_ids = tf.get_default_graph().get_tensor_by_name('label_ids_1:0')
    tensor_segment_ids = tf.get_default_graph().get_tensor_by_name('segment_ids_1:0')
    tensor_outputs = tf.get_default_graph().get_tensor_by_name('loss/Sigmoid:0')

def run(raw_data):

    data = json.loads(raw_data)['data']
    predict_examples = []
    for item in data:
        label = [0] * len(labels) 
        guid = item['id']
        text = item['text']
        predict_examples.append(run_stackoverflow_classifier.InputExample(guid=guid, text_a=text, label=label))

    run_stackoverflow_classifier.file_based_convert_examples_to_features(predict_examples, labels, max_seq_length, tokenizer, predict_file)
    record_iterator = tf.python_io.tf_record_iterator(path=predict_file)
    outputs = []
    for i, string_record in enumerate(record_iterator):
        example = tf.train.Example()
        example.ParseFromString(string_record)
        input_ids = example.features.feature['input_ids'].int64_list.value
        input_mask = example.features.feature['input_mask'].int64_list.value
        label_ids = example.features.feature['label_ids'].int64_list.value
        segment_ids = example.features.feature['segment_ids'].int64_list.value
        result = sess.run(tensor_outputs, feed_dict={
            tensor_input_ids: np.array(input_ids).reshape(-1, max_seq_length),
            tensor_input_mask: np.array(input_mask).reshape(-1, max_seq_length),
            tensor_label_ids: np.array(label_ids).reshape(-1, len(label_ids)),
            tensor_segment_ids: np.array(segment_ids).reshape(-1, max_seq_length),
        })
        output = { 
            'id': data[i]['id'],
            'text': data[i]['text'],
            'probabilities': {}
        }
        for i, tag in enumerate(labels):
            output['probabilities'][tag] = result[0][i]
        outputs.append(output)
        return outputs
    