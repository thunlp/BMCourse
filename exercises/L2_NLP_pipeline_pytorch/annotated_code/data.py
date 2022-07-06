import os
from io import open
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        # 添加一个参数，记录设置的最大序列长度
        self.train = self.tokenize(os.path.join(path, 'train.txt'))  # 改成相应的文件名
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        # self.dictionary.add_word('<pad>')  # 用于padding
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']  # 不再需要<eos>了。为什么？<eos>在LM任务中起到什么作用？
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            ###############################################################################
            # sst2按行构建输入, 长于seq_len的句子进行截断，短于seq_len的用<pad>补齐长度至seq_len
            ###############################################################################
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

            ###############################################################################
            # 构建输出, sst2的label取出, positive的label为1, negative的label为0, 返回格式为多行(ids, label)
            ###############################################################################
        
        ###############################################################################
        # sst2语料库 
        # The dinner is great. \t positive
        # I hate summer. \t negative
        # ...
        # sst2返回格式, 其中pad_id=0
        # (torch.tensor[9,6,7,4,0,0,...], 1)
        # (torch.tensor[3,2,8,0,0,0,...], 0)
        # ...
        ###############################################################################
        return ids
