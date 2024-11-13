import os
import torch
import argparse
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config
from dataset import NERDataset
from utils import *


def calculate_metrics(y_true, y_pred):
    # 计算包含所有标签的结果
    labels = ['B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'O']
    full_report = classification_report(y_true, y_pred, labels=labels)

    # 计算去除 'O' 标签后的结果
    y_true_filtered = [label for label in y_true if label != 'O']
    y_pred_filtered = [label for i, label in enumerate(y_pred) if y_true[i] != 'O']
    labels = ['B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG']
    filtered_report = classification_report(y_true_filtered, y_pred_filtered, labels=labels)

    print("\n全部标签（含 'O' 标签）:\n", full_report)
    print("\n去除 'O' 标签:\n", filtered_report)


def decode_tags(tag_map, tag_sequences):
    id_to_tag = {id_: tag for tag, id_ in tag_map.items()}
    return [[id_to_tag[tag] for tag in seq] for seq in tag_sequences]


def get_predictions(model, dataloader):
    predictions, true_labels = [], []
    for inputs, labels in tqdm(dataloader, desc="Evaluating"):
        _, pred_tags = model(inputs)
        predictions.extend(pred_tags)

        true_labels.extend([seq[seq.gt(0)].tolist() for seq in labels])

    return predictions, true_labels


def load_model(args, vocab, tag_map, model_type):
    if model_type == 'bi-lstm-crf':
        model = build_Bi_RNN_CRF_model(
            args=args,
            vocab_size=len(vocab),
            tag_size=len(tag_map),
            pretrained=os.path.join(args.output_dir, 'best.pth'),
            verbose=True
        )
    elif model_type == 'bert-bi-lstm-crf':
        model = build_BERT_Bi_RNN_CRF_model(
            args=args,
            tag_size=len(tag_map),
            pretrained=os.path.join(args.output_dir, 'best.pth'),
            verbose=True
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    return model


if __name__ == '__main__':
    # 加载配置和路径
    args = get_config()

    # 加载词汇和标签映射
    # vocab = load_vocab(os.path.join(args.data_dir, 'vocab.txt'))
    vocab = load_vocab('/home/yifei/code/bi-lstm-crf/chinese-bert-wwm-ext/vocab.txt')
    tag_map = load_tag(os.path.join(args.data_dir, 'tag.txt'))

    # 加载模型
    model = load_model(args, vocab, tag_map, 'bert-bi-lstm-crf')

    # 准备测试数据集和 DataLoader
    test_dataset = NERDataset(data_dir=os.path.join(args.data_dir, 'test'), vocab=vocab, tag=tag_map)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 获取预测标签和实际标签
    pred_sequences, true_sequences = get_predictions(model, test_loader)

    # 将标签序列展平并解码
    y_true_flat = [tag for seq in decode_tags(tag_map, true_sequences) for tag in seq]
    y_pred_flat = [tag for seq in decode_tags(tag_map, pred_sequences) for tag in seq]

    # 计算和输出指标
    calculate_metrics(y_true_flat, y_pred_flat)
