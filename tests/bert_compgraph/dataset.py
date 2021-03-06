import torch
from torch.utils.data import Dataset
import numpy as np

class SentimentDataHelper(object):
    def __init__(self, train_file, dev_file, test_file, tokenizer):
        print("--- Loading Dataset ---")
        # self.train = SentimentData(train_file, tokenizer)
        # print("--- finished loading train set")
        self.dev = SentimentData(dev_file, tokenizer)
        # print("--- finished loading dev set")
        self.test = SentimentData(test_file, tokenizer)
        # print("--- finished loading test set")

#  Dataset setup inspired by
#  https://huggingface.co/transformers/custom_datasets.html

class SentimentData(Dataset):
    def __init__(self, file_name, tokenizer):
        self.tokenizer = tokenizer
        print("--- Loading sentences from " + file_name)
        raw_x = []
        raw_y = []
        with open(file_name, 'r') as f:
            for line in f:
                pair = line.split('\t')
                raw_x.append(pair[0])
                label = int(pair[1].rstrip())
                raw_y.append(label)

        # tokenizer() returns a BatchEncoding object. See documentation here:
        # https://huggingface.co/transformers/main_classes/tokenizer.html#transformers.BatchEncoding
        self.raw_x = tokenizer(raw_x, truncation=True, padding=True)
        self.raw_y = raw_y
        print(f"--- Loaded {len(raw_x)} sentences from {file_name}")

    def __len__(self):
        return len(self.raw_y)

    def __getitem__(self, i):
        item = {key: torch.tensor(val[i]) for key, val in self.raw_x.items()}
        item["labels"] = torch.tensor(self.raw_y[i])
        return item
