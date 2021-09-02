# coding=utf-8
from flask import Flask, render_template, request
from flask import jsonify
import time
import threading
import torch
from ner_run import predict_ner2json
from conf import args_ner
from utils import label2id_load
from processor import NerProcessor
from ner_run import create_model


app = Flask(__name__,static_url_path="/static")
args = args_ner()
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.label2id = label2id_load()
args.id2label = {v: k for k, v in args.label2id.items()}
processor = NerProcessor()
model, tokenizer = create_model(args, processor)
model.to(args.device)

# 定义心跳检测函数
def heartbeat():
    print (time.strftime('%Y-%m-%d %H:%M:%S - heartbeat', time.localtime(time.time())))
    timer = threading.Timer(60, heartbeat)
    timer.start()
timer = threading.Timer(60, heartbeat)
timer.start()


# 定义应答函数，用于获取输入信息并返回相应的答案
@app.route('/message', methods=['POST'])
def reply():
    sentence = request.form['msg']
    res_msg = predict_ner2json(args, model, tokenizer, sentence)
    print(res_msg[0])
    return jsonify({'text': res_msg[0]['word']})


@app.route("/")
def index():
    return render_template("index.html")


if (__name__ == "__main__"):
    app.run(host='127.0.0.1', port=80)
