import logging
import uuid
import torch
import numpy as np
import KM_parser
tokens = KM_parser
import nltk
# from nltk import word_tokenize, sent_tokenize

uid = uuid.uuid4().hex[:6]

REVERSE_TOKEN_MAPPING = dict([(value, key) for key, value in tokens.BERT_TOKEN_MAPPING.items()])

def torch_load(load_path):
    if KM_parser.use_cuda:
        return torch.load(load_path)
    else:
        return torch.load(load_path, map_location=lambda storage, location: storage)
    
    
class Config(object):
    model_path_base = './LAL-Parser/best_model/best_parser.pt'
    contributions = 0


class ParseHead(object):
    def __init__(self, config) -> None:

        print("Loading model from {}...".format(config.model_path_base))
        assert config.model_path_base.endswith(".pt"), "Only pytorch savefiles supported"

        info = torch_load(config.model_path_base)
        assert 'hparams' in info['spec'], "Older savefiles not supported"
        # logging.Logger.info("LAL-Parse Parameters: {}".format(info['sepc']))  # 打印LAL-Parse解析树的参数
        print("Loading model done!")
        self.parser = KM_parser.ChartParser.from_spec(info['spec'], info['state_dict'])
        self.parser.contributions = (config.contributions == 1)

    def parse_heads(self, sentence):
        # 关闭BatchNormalization和Dropout，以免影响我们的预测结果
        self.parser.eval()
        with torch.no_grad():  # 不去track操作，避免图的构建
            sentence = sentence.strip()
            split_mothod = lambda x: x.split(' ')
            tagged_sentences = [[(REVERSE_TOKEN_MAPPING.get(tag, tag), REVERSE_TOKEN_MAPPING.get(word, word)) for word, tag in nltk.pos_tag(split_mothod(sentence))]]
            # 通过LAL-Parse构建出了Syn语法树
            # head是从tree中得到的，tree是从parse_from_annotations中得到的
            syntree, _, arc = self.parser.parse_batch(tagged_sentences)
            arc_np = np.asarray(arc, dtype='float32')

        return arc_np, syntree


config = Config()
headparser = ParseHead(config)
