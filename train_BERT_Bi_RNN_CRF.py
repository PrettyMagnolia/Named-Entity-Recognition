import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter  # 新增导入
from utils import *
from dataset import NERDataset
from config import get_config


class Trainer:
    def __init__(self, args):
        self.args = args

        vocab = load_vocab('/home/yifei/code/bi-lstm-crf/chinese-bert-wwm-ext/vocab.txt')
        tag = load_tag(os.path.join(args.data_dir, 'tag.txt'))
        self.model = build_BERT_Bi_RNN_CRF_model(args, len(tag), pretrained=None, verbose=True)

        # datasets
        train_ds = NERDataset(data_dir=os.path.join(args.data_dir, 'train'), vocab=vocab, tag=tag)
        val_ds = NERDataset(data_dir=os.path.join(args.data_dir, 'val'), vocab=vocab, tag=tag)

        self.train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        self.val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.device = running_device(self.args.device)
        self.model.to(self.device)

        self.writer = SummaryWriter(log_dir=args.output_dir)

        self.best_val_loss = float('inf')

    def save_model(self):
        model_path = self.args.output_dir
        torch.save(self.model.state_dict(), os.path.join(model_path, 'best.pth'))

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for xb, yb in tqdm(self.val_dl, desc="Evaluating"):
                xb, yb = xb.to(self.device), yb.to(self.device)
                batch_loss = self.model.loss(xb, yb)

                batch_size = len(xb)
                total_loss += batch_loss.item() * batch_size
                total_samples += batch_size

        avg_loss = total_loss / total_samples
        return avg_loss

    def train(self):
        for epoch in range(self.args.num_epoch):
            # train
            self.model.train()
            bar = tqdm(self.train_dl)
            for bi, (xb, yb) in enumerate(bar):
                self.model.zero_grad()
                loss = self.model.loss(xb.to(self.device), yb.to(self.device))
                loss.backward()
                self.optimizer.step()

                self.writer.add_scalar("Loss/Train", loss.item(), epoch * len(self.train_dl) + bi)

            # evaluation
            val_loss = self.evaluate()
            print("Epoch: {}, Validation Loss: {}".format(epoch + 1, val_loss))
            self.writer.add_scalar("Loss/Validation", val_loss, epoch)

            # save model if it's the best one so far
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if self.args.save_model:
                    self.save_model()
                    print("save model(epoch: {}) => {}".format(epoch, self.args.output_dir))


if __name__ == "__main__":
    args = get_config()
    args.output_dir = get_output_dir(args)

    trainer = Trainer(args)

    trainer.train()
