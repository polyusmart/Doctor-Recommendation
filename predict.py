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
from utils.config import init_opts, train_opts, eval_opts, multihead_att_opts

parser = argparse.ArgumentParser('eval MUL-ATT model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
init_opts(parser)
train_opts(parser)
eval_opts(parser)
multihead_att_opts(parser)
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.cuda.set_device(args.gpu)

print(f'Prediction: randome seed {args.seed}, experiment name: {args.name}, run on gpu {args.gpu}')

def main():
    print(f'{args.name} start!')
    print(f'Loadding embeddings from {args.embeddings_path}...')
    with open(f'./{args.embeddings_path}/profile_embeddings.json', 'r', encoding='utf-8') as f:
        profile_embeddings = json.load(f)
    with open(f'./{args.embeddings_path}/q_embeddings.json', 'r', encoding='utf-8') as f:
        query_embeddings = json.load(f)
    with open(f'./{args.embeddings_path}/dialog_embeddings.json', 'r', encoding='utf-8') as f:
        dialogue_embeddings = json.load(f)
    print('Done')
    
    print('Building testing dataset and dataloader...')
    test_set = pd.read_csv(f'./dataset/test.csv', delimiter='\t', encoding='utf-8', dtype={'dr_id': str})
    test_dataset = MulAttDataset(
        'test', test_set, profile_embeddings, query_embeddings, dialogue_embeddings, 
        dr_dialog_sample=args.dr_dialog_sample, neg_sample=args.neg_sample, output=args.output_dir
    )
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    del test_set, test_dataset
    print('Done')
    
    model = MLP(
        args.in_size, 
        args.hidden_size, 
        args.dropout, 
        args.head_num,
        args.add_self_att_on
    ) 
    model_path = f'{args.output_dir}/ckpt/{args.eval_model}'
    model.load_state_dict(torch.load(model_path))
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    
    print(f'{args.name} start prediction...')
    with open(f'{args.output_dir}/test_{args.eval_model}_score.txt', 'w', encoding='utf-8') as score:
        for features, labels in tqdm(test_dataloader):
            with torch.no_grad():
                if torch.cuda.is_available():
                    features = features.cuda()
                    labels = labels.cuda()
                pred_scores = model(features, args.dr_dialog_sample)
                for pred_score in pred_scores:
                    print(pred_score.cpu().detach().numpy().tolist()[0], file = score)

if __name__ == '__main__':
    main()