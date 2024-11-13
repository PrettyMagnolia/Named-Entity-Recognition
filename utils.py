import os
import torch
from model.Bi_RNN_CRF import BiRnnCrf
from model.BERT_Bi_RNN_CRF import BertBiRnnCrf
from torch.nn.utils.rnn import pad_sequence


def running_device(device):
    return device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_output_dir(args):
    base_dir = args.output_dir
    output_file = f'type{args.rnn_type}_lr{args.lr}_hd{args.hidden_dim}'
    output_dir = os.path.join(base_dir, output_file)

    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def build_Bi_RNN_CRF_model(args, vocab_size, tag_size, pretrained=None, verbose=False):
    model = BiRnnCrf(
        vocab_size=vocab_size,
        tagset_size=tag_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_rnn_layers=args.num_rnn_layers,
        rnn=args.rnn_type
    )

    if pretrained is not None and os.path.exists(pretrained):
        state_dict = torch.load(pretrained)
        model.load_state_dict(state_dict)
        if verbose:
            print("load model weights from {}".format(pretrained))
    return model


def build_BERT_Bi_RNN_CRF_model(args, tag_size, pretrained=None, verbose=False):
    model = BertBiRnnCrf(
        tagset_size=tag_size,
        hidden_dim=args.hidden_dim,
        num_rnn_layers=args.num_rnn_layers,
        rnn=args.rnn_type
    )

    if pretrained is not None and os.path.exists(pretrained):
        state_dict = torch.load(pretrained)
        model.load_state_dict(state_dict)
        if verbose:
            print("load model weights from {}".format(pretrained))
    return model


def load_vocab(vocab_file):
    if os.path.exists(vocab_file):
        with open(vocab_file, 'r') as f:
            vocab = [v.strip() for v in f.readlines()]
            vocab_dict = {token: idx for idx, token in enumerate(vocab)}
            return vocab_dict
    else:
        raise FileNotFoundError(f"vocab file {vocab_file} not found, please first run build_vocab")


def load_tag(tag_file):
    if os.path.exists(tag_file):
        with open(tag_file, 'r') as f:
            tags = [tag.strip() for tag in f.readlines()]
            tags_dict = {token: idx for idx, token in enumerate(tags)}
            return tags_dict
    else:
        raise FileNotFoundError(f"tag file {tag_file} not found.")


def build_vocab(corpus_files, vocab_file):
    vocab = set()
    vocab.add('<PAD>')
    for file in corpus_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Corpus file {file} not found")
        with open(file, 'r') as f:
            for line in f:
                for word in line.strip().split():
                    vocab.add(word)

    with open(vocab_file, 'w') as f:
        for word in sorted(vocab):
            f.write(f"{word}\n")

    print(f"Vocabulary saved to {vocab_file}")


def collate_fn(batch):
    sentences, labels = zip(*batch)

    padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)

    return padded_sentences, padded_labels


if __name__ == '__main__':
    corpus_files = [
        '/home/yifei/code/bi-lstm-crf/data/train/corpus.txt',
        '/home/yifei/code/bi-lstm-crf/data/val/corpus.txt'
    ]
    vocab_file = '/home/yifei/code/bi-lstm-crf/data/vocab.txt'
    build_vocab(corpus_files, vocab_file)
