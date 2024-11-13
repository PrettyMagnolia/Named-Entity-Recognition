import torch
import torch.nn as nn
from transformers import BertModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .crf import CRF


class BertBiRnnCrf(nn.Module):
    def __init__(self, tagset_size, hidden_dim, num_rnn_layers=1, rnn="lstm", bert_model_name="/home/yifei/code/bi-lstm-crf/chinese-bert-wwm-ext"):
        super(BertBiRnnCrf, self).__init__()
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size

        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_hidden_size = self.bert.config.hidden_size  # BERT's hidden size
        for name, param in self.bert.named_parameters():
            if "encoder.layer.10" in name or "encoder.layer.11" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # RNN layer after BERT
        RNN = nn.LSTM if rnn == "lstm" else nn.GRU
        self.rnn = RNN(bert_hidden_size, hidden_dim // 2, num_layers=num_rnn_layers,
                       bidirectional=True, batch_first=True)

        # CRF layer for sequence tagging
        self.crf = CRF(hidden_dim, tagset_size)

    def __build_features(self, sentences):
        attention_mask = sentences.gt(0)
        bert_outputs = self.bert(input_ids=sentences, attention_mask=attention_mask)
        embeds = bert_outputs.last_hidden_state  # shape: (batch_size, seq_len, bert_hidden_size)

        # Calculate sequence lengths and masks
        seq_length = attention_mask.sum(1)
        sorted_seq_length, perm_idx = seq_length.sort(descending=True)
        embeds = embeds[perm_idx, :]

        # Apply RNN
        pack_sequence = pack_padded_sequence(embeds, lengths=sorted_seq_length.to('cpu'), batch_first=True)
        packed_output, _ = self.rnn(pack_sequence)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        _, unperm_idx = perm_idx.sort()
        lstm_out = lstm_out[unperm_idx, :]

        return lstm_out, attention_mask

    def loss(self, xs, tags):

        features, masks = self.__build_features(xs)
        loss = self.crf.loss(features, tags, masks=masks)
        return loss

    def forward(self, xs):
        # Get features from BERT and BiLSTM
        features, masks = self.__build_features(xs)
        scores, tag_seq = self.crf(features, masks)
        return scores, tag_seq