import argparse
import sys
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange
from transformers import BertForSequenceClassification, BertTokenizer
sys.path.append('..')
from utils.dataset import NLIDataset

parser = argparse.ArgumentParser()
parser.add_argument('-seed', default=2021, type=int)
parser.add_argument('-epoch_num', default=20, type=int)
parser.add_argument('-batch_size', default=50, type=int)
parser.add_argument('-accumulation_steps', default=5, type=int)
args = parser.parse_args()

# set you available gpus
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
n_gpu = torch.cuda.device_count()
print('gpu num: ', n_gpu)
if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

def create_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    
    if samples[0][2] is not None:
        label_ids = torch.stack([s[2] for s in samples])
    else:
        label_ids = None
    
    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)
    
    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)
    
    return tokens_tensors, segments_tensors, masks_tensors, label_ids

def get_predictions(model, dataloader, compute_acc=False):
    predictions = None
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(dataloader):
            data = [t.to(device) for t in data if t is not None]
            tokens_tensors, segments_tensors, masks_tensors = data[:3]
            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors)
            logits = outputs[0]
            _, pred = torch.max(logits.data, 1)
            if compute_acc:
                labels = data[3]
                total += labels.size(0)
                correct += (pred == labels).sum().item()
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))
    if compute_acc:
        acc = correct / total
        return predictions, acc
    return predictions

def train(model, optimizer, trainloader, validloader, device):
    for epoch in trange(0, args.epoch_num):
        print('\nEpoch: ', epoch)
        model.train()
        running_loss = []
        i = 0
        for data in tqdm(trainloader):
            tokens_tensors, segments_tensors, masks_tensors, labels = [t.to(device) for t in data]
            optimizer.zero_grad() 
            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors, 
                            labels=labels)
            loss = outputs[0] 
            if n_gpu > 1:
                loss = loss.mean()
            loss = loss / args.accumulation_steps
            loss.backward()
            running_loss.append(loss.item())
            if (i+1) % args.accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()
            i = i + 1
        _, train_acc = get_predictions(model, trainloader, compute_acc=True)
        print(f'Epoch: {epoch}, train loss: {np.mean(running_loss)}, train classification acc: {train_acc}')
        if not os.path.exists(f'./checkpoints'):
            os.makedirs(f'./checkpoints')
        if n_gpu > 1:
            model_state_dict = model.module.bert.state_dict()
        else:
            model_state_dict = model.bert.state_dict()
        torch.save(model_state_dict, f'./checkpoints/model_{epoch}.bin')
        model.eval()
        _, valid_acc = get_predictions(model, validloader, compute_acc=True)
        print(f'Epoch: {epoch}, valid classification acc: {valid_acc}')

def main():
    tokenizer = BertTokenizer.from_pretrained('../mc_bert_base/')
    model = BertForSequenceClassification.from_pretrained('../mc_bert_base/', num_labels=2)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    trainset = NLIDataset('train', tokenizer=tokenizer)
    validset = NLIDataset('valid', tokenizer=tokenizer)
    testset = NLIDataset('test', tokenizer=tokenizer)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, collate_fn=create_batch)
    validloader = DataLoader(validset, batch_size=args.batch_size, collate_fn=create_batch)
    testloader = DataLoader(testset, batch_size=args.batch_size, collate_fn=create_batch)
    
    _, test_acc = get_predictions(model, testloader, compute_acc=True)
    print(f'Model without training classification acc: {test_acc}')
    # model training
    print('Training model...')
    train(model, optimizer, trainloader, validloader, device)
    print('Model training done!')
    # model testing
    print('Testing model...')
    _, test_acc = get_predictions(model, testloader, compute_acc=True)
    print(f'Model testing done with classification acc: {test_acc}')
    
if __name__ == '__main__':
    main()
