import glob
import math
import os
import random
import time
import cv2
import numpy as np
import pandas as pd
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
    model = 'densenet121'
    #     model = 'resnet34d'
    train_bs = 16
    valid_bs = 16
    epochs = 40
    lr = 1e-3
    weight_decay = 0
    img_size = 512
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
# test_df = pd.read_csv('test.csv', header=None, names=['filename','s/e','time'])
test_df = pd.read_csv('test_b.csv', header=None, names=['filename', 's/e', 'time'])

train_df.loc[train_df['s/e'] == 's', 'label'] = train_df[train_df['s/e'] == 's'].time / 200
train_df.loc[train_df['s/e'] == 'e', 'label'] = train_df[train_df['s/e'] == 'e'].time / 180

test_df['label'] = 0


class MyDataset(Dataset):
    def __init__(self, df, length=200):
        self.df = df
        self.length = length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df.filename.values[idx]

        image = np.load('spectrogram/' + filename.split('.')[0] + '.npy')

        label = self.df.label.values[idx]

        return image, label


test_set = MyDataset(test_df[test_df['s/e'] == 's'])
test_loader = DataLoader(test_set, batch_size=CFG.valid_bs, shuffle=False, num_workers=CFG.num_workers)


class Model(nn.Module):
    def __init__(self, base_model_name: str, num_classes=1, in_channels=3):
        super().__init__()
        self.encoder = timm.create_model(base_model_name, pretrained=True, in_chans=in_channels,
                                         num_classes=num_classes)

    def forward(self, x):
        x = self.encoder(x)  # (bs, len, 1)
        x = torch.sigmoid(x).squeeze(-1)
        return x


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


def train_model(model, train_loader):
    model.train()

    losses = AverageMeter()

    optimizer.zero_grad()

    tk = tqdm(train_loader, total=len(train_loader), position=0, leave=True)
    for step, batch in enumerate(tk):
        image, label = batch
        image, label = image.to(device), label.to(device).float()

        with autocast():
            output = model(image)
            loss = criterion(output, label)

        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        lr = optimizer.param_groups[-1]['lr']

        losses.update(loss.item(), label.size(0))
        tk.set_postfix(loss=losses.avg, lr=lr)

    return losses.avg


def test_model(model, val_loader):
    model.eval()

    losses = AverageMeter()
    y_true, y_pred = [], []

    with torch.no_grad():
        tk = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
        for step, batch in enumerate(tk):
            image, label = batch
            image, label = image.to(device), label.to(device).float()

            output = model(image)

            loss = criterion(output, label)

            losses.update(loss.item(), label.size(0))
            tk.set_postfix(loss=losses.avg)

    return losses.avg


data = train_df[train_df['s/e'] == 's'].reset_index()

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

    best_loss = 1e6
    steps_per_epoch = len(train_loader)

    model = Model(CFG.model).to(device)
    model.load_state_dict(torch.load('models/{}_fold_{}_start.pt'.format(CFG.model, fold)))

    scaler = GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    criterion = nn.L1Loss()
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0.05 * CFG.epochs * steps_per_epoch,
                                                CFG.epochs * steps_per_epoch)

    for epoch in range(CFG.epochs):
        print('epoch:', epoch)
        time.sleep(0.2)

        train_loss = train_model(model, train_loader)
        val_loss = test_model(model, val_loader)

        if np.isnan(val_loss):
            break

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), '{}_fold_{}_start.pt'.format(CFG.model, fold))

    cvstart.append(best_loss)

print(cvstart, np.mean(cvstart) * 200)

test_set = MyDataset(test_df[test_df['s/e'] == 's'])
test_loader = DataLoader(test_set, batch_size=CFG.valid_bs, shuffle=False, num_workers=CFG.num_workers)

model = Model(CFG.model).to(device)
pred = []

for i in range(5):
    model.load_state_dict(torch.load('{}_fold_{}_start.pt'.format(CFG.model, i)))
    model.eval()

    outputs = []

    with torch.no_grad():
        tk = tqdm(test_loader, total=len(test_loader), position=0, leave=True)
        for step, batch in enumerate(tk):
            image, label = batch
            image, label = image.to(device), label.to(device).float()

            output = model(image).cpu().numpy() * 200

            outputs.extend(output)

    pred.append(outputs)

pred = np.array(pred)  # (5, 350)
ensemble_pred = []

for i in range(pred.shape[1]):
    ensemble_pred.append((pred[:, i].sum() - pred[:, i].max() - pred[:, i].min()) / 3)

pred = ensemble_pred
test_df.loc[test_df['s/e'] == 's', 'time'] = pred
test_df['time'] = test_df.time.apply(lambda x: round(x, 3))

data = train_df[train_df['s/e'] == 'e'].reset_index()

seed_everything(CFG.seed)

folds = StratifiedKFold(n_splits=CFG.fold_num, shuffle=True, random_state=CFG.seed).split(np.arange(data.shape[0]), (
    data['label'].values) * 100 // 10)

cvend = []

for fold, (trn_idx, val_idx) in enumerate(folds):

    print(fold)

    train = data.loc[trn_idx]
    val = data.loc[val_idx]

    train_set = MyDataset(train)
    val_set = MyDataset(val)

    train_loader = DataLoader(train_set, batch_size=CFG.train_bs, shuffle=True, num_workers=CFG.num_workers)
    val_loader = DataLoader(val_set, batch_size=CFG.valid_bs, shuffle=False, num_workers=CFG.num_workers)

    best_loss = 1e6
    steps_per_epoch = len(train_loader)

    model = Model(CFG.model).to(device)
    model.load_state_dict(torch.load('models/{}_fold_{}_end.pt'.format(CFG.model, fold)))

    scaler = GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    criterion = nn.L1Loss()
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0.05 * CFG.epochs * steps_per_epoch,
                                                CFG.epochs * steps_per_epoch)

    for epoch in range(CFG.epochs):
        print('epoch:', epoch)
        time.sleep(0.2)

        train_loss = train_model(model, train_loader)
        val_loss = test_model(model, val_loader)

        if np.isnan(val_loss):
            break

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), '{}_fold_{}_end.pt'.format(CFG.model, fold))

    cvend.append(best_loss)

print(cvend, np.mean(cvend) * 180)

test_set = MyDataset(test_df[test_df['s/e'] == 'e'], 180)
test_loader = DataLoader(test_set, batch_size=CFG.valid_bs, shuffle=False, num_workers=CFG.num_workers)

model = Model(CFG.model).to(device)
pred = []

for i in range(5):
    model.load_state_dict(torch.load('{}_fold_{}_end.pt'.format(CFG.model, i)))
    model.eval()

    outputs = []

    with torch.no_grad():
        tk = tqdm(test_loader, total=len(test_loader), position=0, leave=True)
        for step, batch in enumerate(tk):
            image, label = batch
            image, label = image.to(device), label.to(device).float()

            output = model(image).cpu().numpy() * 180

            outputs.extend(output)

    pred.append(outputs)

pred = np.array(pred)  # (5, 350)
ensemble_pred = []

for i in range(pred.shape[1]):
    ensemble_pred.append((pred[:, i].sum() - pred[:, i].max() - pred[:, i].min()) / 3)

pred = ensemble_pred
test_df.loc[test_df['s/e'] == 'e', 'time'] = pred
test_df['time'] = test_df.time.apply(lambda x: round(x, 3))

test_df[['filename', 's/e', 'time']].to_csv('submission.csv', index=False, header=None)
