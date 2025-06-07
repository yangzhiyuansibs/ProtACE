# import
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import optuna
from time import sleep
import itertools
import numpy as np
from sklearn import metrics
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic=True
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
##k-mer encoding
def encode_sequence(seq):
    k=3
    kmer_dict = {''.join(i): idx for idx, i in enumerate(itertools.product('ACDEFGHIKLMNPQRSTVWY', repeat=k))}
    encoding = torch.zeros(len(kmer_dict), dtype=torch.float32)
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        encoding[kmer_dict[kmer]] += 1

    return encoding
##load_file function demo: load excel data
def load_data(file, label):
  df = pd.read_excel(file)
  sequences = df.iloc[:, 1]
  features = []
  for seq in sequences:
    features.append(encode_sequence(seq))
  features = torch.stack(features).unsqueeze(1)
  labels = torch.tensor([label] * len(sequences), dtype=torch.long)
  return features, labels


####
#Attention module
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.linear = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, lstm_out):
        out = self.linear(lstm_out)
        score = torch.bmm(out, out.transpose(1, 2))
        attn = self.softmax(score)

        context = torch.bmm(attn, lstm_out)
        return context
###
#LSTM-Att
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,drop):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(drop)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True)
        self.attention = Attention(hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        # x shape (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        out = self.attention(out)
        out = out.permute(0, 2, 1)
        out = self.batch_norm(out)
        out = out.permute(0, 2, 1)
        out = self.fc(out[:, -1, :])
        return out
##CNN-Att
class CNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,drop):
        super(CNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(drop)
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.attention = Attention(hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        # x shape (batch, seq_len, input_size)
        out = x.permute(0, 2, 1)
        out = self.conv(out)
        out = out.permute(0, 2, 1)
        out = self.attention(out)
        out = out.permute(0, 2, 1)
        out = self.batch_norm(out)
        out = out.permute(0, 2, 1)
        out = self.fc(out[:, -1, :])
        return out



##train
epochs = 122
for epoch in range(epochs):
    
    model.train()

    outputs = model(train_features)

    loss = criterion(outputs, train_labels)

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()
    train_preds = torch.argmax(outputs, dim=1)
    train_acc = torch.sum(train_preds == train_labels).item() / len(train_labels)
    model.eval()
    val_loss = criterion(val_outputs, test_labels).item()
    val_preds = torch.argmax(val_outputs, dim=1)
    val_acc = torch.sum(val_preds == test_labels).item() / len(test_labels)\
##eval on test set
model.eval()
outputs = model(test_set)
loss = criterion(outputs, test_lab)
test_loss = criterion(outputs, test_lab).item()
preds = torch.argmax(outputs, dim=1)
