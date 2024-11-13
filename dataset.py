import os
import json
import torch
from torch.utils.data import Dataset
from transformers.utils.logging import tqdm


class NERDataset(Dataset):
    def __init__(self, data_dir, vocab, tag):
        self.data_dir = data_dir

        self.corpus_file = os.path.join(data_dir, 'corpus.txt')
        self.label_file = os.path.join(data_dir, 'label.txt')

        self.vocab = vocab
        self.tag = tag

        self.OOV_IDX = vocab['[UNK]']

        self.sentences, self.labels = self.load_dataset()

    def load_dataset(self):
        sentences = []
        labels = []

        with open(self.corpus_file, 'r', encoding='utf-8') as f_text, open(self.label_file, 'r',
                                                                           encoding='utf-8') as f_label:
            for text_line, label_line in tqdm(zip(f_text, f_label)):
                words = text_line.strip().split()
                tags = label_line.strip().split()

                # 将词和标签转换为对应的id
                word_ids = [self.vocab.get(word, self.OOV_IDX) for word in words]
                tag_ids = [self.tag[tag] for tag in tags]

                sentences.append(word_ids)
                labels.append(tag_ids)

        return sentences, labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return torch.tensor(self.sentences[idx]), torch.tensor(self.labels[idx])
