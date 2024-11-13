# Named Entity Recognition (NER)

## 1. Pretrained Model
A pretrained model can be found on Hugging Face: [Download here](https://huggingface.co/BranLiu/BERT-Bi-LSTM-CRF).

## 2. Chinese BERT Model
For the BERT-based NER model, the [chinese-bert-wwm-ext](https://huggingface.co/hfl/chinese-bert-wwm-ext) model from Hugging Face is utilized as the initial embedding layer. This model supports whole-word masking, enhancing its effectiveness for Chinese NER tasks.

## 3. Training
To train the models, use the following scripts:
- `train_Bi_RNN-CRF.py` for training the Bi-RNN-CRF model.
- `train_BERT_Bi_RNN_CRF.py` for training the BERT-Bi-RNN-CRF model.

## 4. Evaluation
Use the `test.py` script to evaluate the model's performance and calculate Precision (P), Recall (R), and F1-score (F1).