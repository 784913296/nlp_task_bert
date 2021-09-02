#!/usr/bin/env python
# _*_coding:utf-8_*_
import json
import time
import logging
import os
from flask import Flask, request
from task_classes.conf import Args
from task_classes.processor import classesProcessor
from task_classes.predict import one_text_predict
from utils.util import filterContent
from models.model import create_model, create_tokenizer, last_model_file


logger = logging.getLogger(__name__)
args = Args().get_parser()
processor = classesProcessor()


def predict(model, tokenizer, text, rawText, msgId, userId, botId):
    """ 给定文本, 返回分类id """
    ret_dict = {}
    index2cid = processor.get_index2cid()
    cid2name = processor.get_cid2name()
    start = time.time()
    pred = one_text_predict(text, model, tokenizer, index2cid)

    ret_dict['msgId'] = msgId
    ret_dict['content'] = rawText
    ret_dict['userId'] = userId
    ret_dict['botId'] = botId
    ret_dict['knowledgeId'] = pred
    ret_dict['knowledgeName'] = cid2name[pred]
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
    content = request.args.get('content')
    msgId = request.args.get('msgId')
    userId = request.args.get('userId')
    botId = request.args.get('botId')
    if content != None:
        text = text_preprocess.word_segment(content)
        ret = predict(model, tokenizer, text, content, msgId, userId, botId)
    return json.dumps(ret, ensure_ascii=False)


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=19000)
