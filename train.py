import argparse
import json
from collections import Counter
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from Modules.MLP import MLP
from utils.dataset import MulAttDataset
from utils.EarlyStopping import EarlyStopping
from utils.loss import weighted_class_bceloss
from utils.config import init_opts, train_opts, multihead_att_opts

parser = argparse.ArgumentParser('train MUL-ATT model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
init_opts(parser)
train_opts(parser)
multihead_att_opts(parser)
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.cuda.set_device(args.gpu)

print(f'Training: randome seed {args.seed}, experiment name: {args.name}, run on gpu {args.gpu}')

def train_model(model, train_dataloader, val_dataloader):
    print(f'{args.name} start training...')
    weights = torch.Tensor([1, 5])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=args.patience, threshold=1e-4, min_lr=1e-5)
    early_stopping = EarlyStopping(patience = args.patience, verbose = True, path = f'{args.output_dir}/ckpt/best_model.pt')
    
    if torch.cuda.is_available():
        model = model.cuda()
        weights = weights.cuda()

    train_losses = []
    valid_losses = []
    for epoch in trange(args.epoch_num):
        print(f'Current Epoch: {epoch+1}')
        model.train()
        for features, labels in tqdm(train_dataloader):
            if torch.cuda.is_available():
                features = features.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            pred_scores = model(features, args.dr_dialog_sample)
            loss = weighted_class_bceloss(pred_scores, labels.reshape(-1, 1), weights)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0, norm_type=2)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            for features, labels in tqdm(val_dataloader):
                if torch.cuda.is_available():
                    features = features.cuda()
                    labels = labels.cuda()
                pred_scores = model(features, args.dr_dialog_sample)
                loss = weighted_class_bceloss(pred_scores, labels.reshape(-1, 1), weights)
                valid_losses.append(loss.item())
                
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        print(f'\nEpoch {epoch+1}, train loss: {train_loss}, valid loss: {valid_loss}')
        torch.save(model.state_dict(),f'./{args.output_dir}/ckpt/model_{epoch+1}.pt')
        if (epoch+1 > 15): 
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print(f'Early stopping at epoch {epoch+1}')
                break  
        scheduler.step(valid_loss)

def main():
    print(f'Loadding embeddings from {args.embeddings_path}...')
    with open(f'./{args.embeddings_path}/profile_embeddings.json', 'r', encoding='utf-8') as f:
        profile_embeddings = json.load(f)
    with open(f'./{args.embeddings_path}/q_embeddings.json', 'r', encoding='utf-8') as f:
        query_embeddings = json.load(f)
    with open(f'./{args.embeddings_path}/dialog_embeddings.json', 'r', encoding='utf-8') as f:
        dialogue_embeddings = json.load(f)
    print('Done')
    
    print('Building training dataset and dataloader...')
    train_set = pd.read_csv(f'./dataset/train.csv', delimiter='\t', encoding='utf-8', dtype={'dr_id': str})
    train_dataset = MulAttDataset(
        'train', train_set, profile_embeddings, query_embeddings, dialogue_embeddings, 
        dr_dialog_sample=args.dr_dialog_sample, neg_sample=args.neg_sample
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    del train_set, train_dataset
    print('Done')
    
    print('Building validation dataset and dataloader...')
    valid_set = pd.read_csv(f'./dataset/valid.csv', delimiter='\t', encoding='utf-8', dtype={'dr_id': str})
    val_dataset = MulAttDataset(
        'valid', valid_set, profile_embeddings, query_embeddings, dialogue_embeddings, 
        dr_dialog_sample=args.dr_dialog_sample, neg_sample=args.neg_sample
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    del valid_set, val_dataset, profile_embeddings, query_embeddings, dialogue_embeddings
    print('Done')
    
    model = MLP(
        args.in_size, 
        args.hidden_size, 
        args.dropout, 
        args.head_num,
        args.add_self_att_on
    ) 
    
    train_model(model, train_dataloader, val_dataloader)

if __name__ == '__main__':
    main()