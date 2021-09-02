#!/usr/bin/env python
# _*_coding:utf-8_*_
import json
import time
import logging
import os
from flask import Flask, request
from task_sim.conf import args_sim
from task_sim.processor import simProcessor
from task_sim.predict import pred
from utils.util import filterContent
from models.model import create_model, create_tokenizer, last_model_file


logger = logging.getLogger(__name__)
args = args_sim()
processor = simProcessor()
args.num_labels = len(processor.get_labels())


def predict(model, text_a, text_b, msgId, userId, botId):
    """ 给定文本, 返回分类id """
    ret_dict = {}
    id2lable = {0: '不相似', 1: '相似'}
    start = time.time()
    pred_label = pred(args, model, text_a, text_b)

    ret_dict['msgId'] = msgId
    ret_dict['text_a'] = text_a
    ret_dict['text_b'] = text_b
    ret_dict['userId'] = userId
    ret_dict['botId'] = botId
    ret_dict['labelId'] = pred_label
    ret_dict['labelName'] = id2lable[pred_label]
    end = time.time()
    logger.info('inference time: {}'.format(end - start))
    return ret_dict
    # return json.dumps(ret_dict, ensure_ascii=False)


app = Flask(__name__)
text_preprocess = filterContent()
tokenizer = create_tokenizer(args)
output_dir = os.path.join(args.output_dir, args.bert_type, args.task_type)
model_file = last_model_file(output_dir, count=1)[0]
model = create_model(args, model_file)
model.eval()


@app.route('/query')
def query():
    ret = []
    text_a = request.args.get('text_a')
    text_b = request.args.get('text_b')
    msgId = request.args.get('msgId')
    userId = request.args.get('userId')
    botId = request.args.get('botId')
    if text_a != None:
        text_a = text_preprocess.word_segment(text_a)
        text_b = text_preprocess.word_segment(text_b)

        ret = predict(model, text_a, text_b, msgId, userId, botId)
    return json.dumps(ret, ensure_ascii=False)


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=19000)
