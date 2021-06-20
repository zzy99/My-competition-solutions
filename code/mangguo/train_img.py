import glob
import math
import os
import random
import cv2
import numpy as np
import pandas as pd
import scipy.stats as stats
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import *
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import *


class CFG:
    fold_num = 5
    seed = 42
    train_bs = 32
    valid_bs = 32
    epochs = 50
    lr = 1e-3
    weight_decay = 0
    num_workers = 0
    device = 0


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(CFG.seed)

torch.cuda.set_device(CFG.device)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv', header=None, names=['filename', 's/e', 'time'])
test_b_df = pd.read_csv('test_b.csv', header=None, names=['filename', 's/e', 'time'])

train_df.loc[train_df['s/e'] == 's', 'label'] = train_df[train_df['s/e'] == 's'].time / 200
train_df.loc[train_df['s/e'] == 'e', 'label'] = train_df[train_df['s/e'] == 'e'].time / 180

test_df['label'] = 0
test_b_df['label'] = 0


class MyDataset(Dataset):
    def __init__(self, df, length=200):
        self.df = df
        self.length = length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df.filename.values[idx]

        image = np.load('frames_npy/' + filename.split('.')[0] + '.npy')

        label = np.zeros((image.shape[0],))
        time = math.floor(self.df.time.values[idx])
        label[time] = 0.8
        label[time - 1:time + 2:2] = 0.1

        label2 = np.zeros((image.shape[0] * 10,))
        time = math.floor(self.df.time.values[idx] * 10)
        for i in np.arange(-5, 6, 1):
            if 0 <= time + i < image.shape[0] * 10:
                label2[time + i] = stats.norm.pdf(i, loc=0, scale=2)

        time = self.df.time.values[idx]

        return image, label, label2, time


class Model(nn.Module):
    def __init__(self, num_classes=1, length=200):
        super().__init__()
        hidden = 256
        self.rnn = nn.GRU(512, hidden, 2, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden * 2, num_classes)
        self.fc2 = nn.Linear(hidden * 2, num_classes * 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):  # (bs, len, 512)
        bs = x.size(0)
        x = self.dropout(x)
        out, hidden = self.rnn(x)  # (bs, len, hidden*2)
        y = self.fc1(out).squeeze(-1)  # (bs, len)
        y2 = self.fc2(out).view(bs, -1)  # (bs, len*10)
        y3 = (y.softmax(1).unsqueeze(2).repeat(1, 1, 10) / 10).view(bs, -1) + y2.softmax(1)  # (bs, len*10)
        return y, y2, y3


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_model(model, train_loader, epoch):
    model.train()

    losses = AverageMeter()

    optimizer.zero_grad()

    tk = tqdm(train_loader, total=len(train_loader), position=0, leave=True)
    for step, batch in enumerate(tk):
        image, label, label2, time = batch
        image, label, label2, time = image.to(device), label.to(device), label2.to(device), time.to(device).float()

        with autocast():
            output, output2, output3 = model(image)
            loss = criterion(output, label.argmax(1).long()) + criterion(output2, label2.argmax(1).long()) * 0.1
        #                 + criterion(output3, label2.argmax(1).long())*0.1
        # + nn.L1Loss()(output.argmax(1), label.argmax(1).float()) + nn.L1Loss()(offset, time-label.argmax(1))*0.1

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        lr = optimizer.param_groups[-1]['lr']

        losses.update(loss.item(), label.size(0))
        tk.set_postfix(loss=losses.avg, lr=lr)
        scheduler.step()
    return losses.avg


def test_model(model, val_loader):
    model.eval()

    losses = AverageMeter()
    scores = AverageMeter()
    scores2 = AverageMeter()
    scores3 = AverageMeter()

    with torch.no_grad():
        tk = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
        for step, batch in enumerate(tk):
            image, label, label2, time = batch
            image, label, label2, time = image.to(device), label.to(device), label2.to(device), time.to(device).float()

            output, output2, output3 = model(image)
            loss = criterion(output, label.argmax(1).long())
            score = abs(output.argmax(1) + 0.5 - time).float().mean()
            score2 = abs(output2.argmax(1) / 10 + 0.05 - time).float().mean()
            score3 = abs(output3.argmax(1) / 10 + 0.05 - time).float().mean()

            losses.update(loss.item(), label.size(0))
            scores.update(score.item(), label.size(0))
            scores2.update(score2.item(), label.size(0))
            scores3.update(score3.item(), label.size(0))

            tk.set_postfix(loss=losses.avg, score1=scores.avg, score2=scores2.avg, score3=scores3.avg)

    return losses.avg, scores.avg, scores2.avg, scores3.avg


data = train_df[train_df['s/e'] == 's'].reset_index()

seed_everything(CFG.seed)

folds = StratifiedKFold(n_splits=CFG.fold_num, shuffle=True, random_state=CFG.seed).split(np.arange(data.shape[0]), (
    data['label'].values) * 100 // 10)

cvstart = []

for fold, (trn_idx, val_idx) in enumerate(folds):

    print(fold)

    train = data.loc[trn_idx]
    val = data.loc[val_idx]

    train_set = MyDataset(train)
    val_set = MyDataset(val)

    train_loader = DataLoader(train_set, batch_size=CFG.train_bs, shuffle=True, num_workers=CFG.num_workers)
    val_loader = DataLoader(val_set, batch_size=CFG.valid_bs, shuffle=False, num_workers=CFG.num_workers)

    best_score = 1e6
    steps_per_epoch = len(train_loader)

    model = Model().to(device)

    scaler = GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0.05 * CFG.epochs * steps_per_epoch,
                                                CFG.epochs * steps_per_epoch)

    for epoch in range(CFG.epochs):
        print('epoch:', epoch)
        import time

        time.sleep(0.2)

        train_loss = train_model(model, train_loader, epoch)
        val_loss, _, _, val_score = test_model(model, val_loader)

        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), '{}_fold_{}_start.pt'.format('GRU', fold))

    cvstart.append(best_score)

print(cvstart, np.mean(cvstart))

data = train_df[train_df['s/e'] == 'e'].reset_index()

seed_everything(CFG.seed)

folds = StratifiedKFold(n_splits=CFG.fold_num, shuffle=True, random_state=CFG.seed).split(np.arange(data.shape[0]), (
    data['label'].values) * 100 // 10)

cvend = []

for fold, (trn_idx, val_idx) in enumerate(folds):

    print(fold)

    train = data.loc[trn_idx]
    val = data.loc[val_idx]

    train_set = MyDataset(train, 180)
    val_set = MyDataset(val, 180)

    train_loader = DataLoader(train_set, batch_size=CFG.train_bs, shuffle=True, num_workers=CFG.num_workers)
    val_loader = DataLoader(val_set, batch_size=CFG.valid_bs, shuffle=False, num_workers=CFG.num_workers)

    best_score = 1e6
    steps_per_epoch = len(train_loader)

    model = Model().to(device)

    scaler = GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0.05 * CFG.epochs * steps_per_epoch,
                                                CFG.epochs * steps_per_epoch)

    for epoch in range(CFG.epochs):
        print('epoch:', epoch)
        import time

        time.sleep(0.2)

        train_loss = train_model(model, train_loader, epoch)
        val_loss, _, _, val_score = test_model(model, val_loader)

        if np.isnan(val_loss):
            break

        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), '{}_fold_{}_end.pt'.format('GRU', fold))

    cvend.append(best_score)

print(cvend, np.mean(cvend))

# # A榜提交
test_set = MyDataset(test_df[test_df['s/e'] == 's'])
test_loader = DataLoader(test_set, batch_size=CFG.valid_bs, shuffle=False, num_workers=CFG.num_workers)

model = Model().to(device)
pred = []
offsets = []

for i in range(5):
    model.load_state_dict(torch.load('{}_fold_{}_start.pt'.format('GRU', i)), strict=False)
    model.eval()

    outputs = []

    with torch.no_grad():
        tk = tqdm(test_loader, total=len(test_loader), position=0, leave=True)
        for step, batch in enumerate(tk):
            image, label, label2, time = batch
            image, label, label2, time = image.to(device), label.to(device), label2.to(device), time.to(device).float()

            output, output2, output3 = model(image)
            output = output3.softmax(1).cpu().numpy()
            outputs.extend(output)

    pred.append(outputs)

pred = np.mean(pred, 0)

test_df.loc[test_df['s/e'] == 's', 'time'] = pred.argmax(1) / 10 + 0.05
test_df['time'] = test_df.time.apply(lambda x: round(x, 3))

test_set = MyDataset(test_df[test_df['s/e'] == 'e'], 180)
test_loader = DataLoader(test_set, batch_size=CFG.valid_bs, shuffle=False, num_workers=CFG.num_workers)

model = Model().to(device)
pred = []
offsets = []

for i in range(5):
    model.load_state_dict(torch.load('{}_fold_{}_end.pt'.format('GRU', i)), strict=False)
    model.eval()

    outputs = []

    with torch.no_grad():
        tk = tqdm(test_loader, total=len(test_loader), position=0, leave=True)
        for step, batch in enumerate(tk):
            image, label, label2, time = batch
            image, label, label2, time = image.to(device), label.to(device), label2.to(device), time.to(device).float()

            output, output2, output3 = model(image)
            output = output3.softmax(1).cpu().numpy()
            outputs.extend(output)

    pred.append(outputs)

pred = np.mean(pred, 0)

test_df.loc[test_df['s/e'] == 'e', 'time'] = pred.argmax(1) / 10 + 0.05  # offsets
test_df['time'] = test_df.time.apply(lambda x: round(x, 3))

test_df[['filename', 's/e', 'time']].to_csv('submission.csv', index=False, header=None)

# # B榜提交
test_set = MyDataset(test_b_df[test_b_df['s/e'] == 's'])
test_loader = DataLoader(test_set, batch_size=CFG.valid_bs, shuffle=False, num_workers=CFG.num_workers)

model = Model().to(device)
pred = []
offsets = []

for i in range(5):
    model.load_state_dict(torch.load('{}_fold_{}_start.pt'.format('GRU', i)), strict=False)
    model.eval()

    outputs = []

    with torch.no_grad():
        tk = tqdm(test_loader, total=len(test_loader), position=0, leave=True)
        for step, batch in enumerate(tk):
            image, label, label2, time = batch
            image, label, label2, time = image.to(device), label.to(device), label2.to(device), time.to(device).float()

            output, output2, output3 = model(image)
            output = output3.softmax(1).cpu().numpy()
            outputs.extend(output)

    pred.append(outputs)

pred = np.mean(pred, 0)

test_b_df.loc[test_b_df['s/e'] == 's', 'time'] = pred.argmax(1) / 10 + 0.05
test_b_df['time'] = test_b_df.time.apply(lambda x: round(x, 3))

test_set = MyDataset(test_b_df[test_b_df['s/e'] == 'e'], 180)
test_loader = DataLoader(test_set, batch_size=CFG.valid_bs, shuffle=False, num_workers=CFG.num_workers)

model = Model().to(device)
pred = []
offsets = []

for i in range(5):
    model.load_state_dict(torch.load('{}_fold_{}_end.pt'.format('GRU', i)), strict=False)
    model.eval()

    outputs = []

    with torch.no_grad():
        tk = tqdm(test_loader, total=len(test_loader), position=0, leave=True)
        for step, batch in enumerate(tk):
            image, label, label2, time = batch
            image, label, label2, time = image.to(device), label.to(device), label2.to(device), time.to(device).float()

            output, output2, output3 = model(image)
            output = output3.softmax(1).cpu().numpy()
            outputs.extend(output)

    pred.append(outputs)

pred = np.mean(pred, 0)

test_b_df.loc[test_b_df['s/e'] == 'e', 'time'] = pred.argmax(1) / 10 + 0.05
test_b_df['time'] = test_b_df.time.apply(lambda x: round(x, 3))

test_b_df[['filename', 's/e', 'time']].to_csv('submission.csv', index=False, header=None)
