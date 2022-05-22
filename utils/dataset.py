from random import sample
from collections import Counter
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class MulAttDataset(Dataset):
    def __init__(self, split, dataset, profile_embeddings, query_embeddings, dialogue_embeddings, 
                 dr_dialog_sample=100, neg_sample=10, embed_size=768, output=''):
        self.split = split
        self.dataset = dataset
        self.q_list = dataset.dialog_id.tolist() # query id - the same as dialogue id 
        self.dr_list = list(set(dataset.dr_id.tolist()))
        self.q_dr_match = dict(zip(dataset.dialog_id, dataset.dr_id))
        self.profile_emb = profile_embeddings
        self.q_emb = query_embeddings
        self.dialog_emb = dialogue_embeddings
        train_set = pd.read_csv('./dataset/train.csv', delimiter='\t', encoding='utf-8', dtype={'dr_id':str})
        self.most_common_drs = [dr for dr, _ in Counter(train_set.dr_id.tolist()).most_common()]
        self.train_q_dr_match = dict(zip(train_set.dialog_id, train_set.dr_id))
        del train_set
        self.dr_dialog_sample = dr_dialog_sample
        self.neg_sample = neg_sample
        self.embed_size = embed_size
        self.output = output
        self.dr_feature = {}
        self.features = []
        self.labels = []
        for dr in tqdm(self.dr_list, desc='packing doctor features'):
            self.pack_dr_features(dr)
        self.pack_dataset()

    def __getitem__(self, index):
        return torch.FloatTensor(self.features[index]), torch.FloatTensor([self.labels[index]])[0]

    def __len__(self):
        return len(self.labels)

    def pack_dr_features(self, dr_id):
        feature = []
        feature_profile = self.profile_emb[dr_id]
        feature.append(feature_profile)
        records = [dialog_id for (dialog_id, doctor_id) in self.train_q_dr_match.items() if doctor_id == dr_id]
        if len(records) > self.dr_dialog_sample:
            sample_records = sample(records, self.dr_dialog_sample)
            for idx in sample_records:
                feature.append(self.dialog_emb[idx])
        else:
            pad_size = self.dr_dialog_sample - len(records)
            for idx in records:
                feature.append(self.dialog_emb[idx])
            feature.extend([[0] * self.embed_size] * pad_size)
        self.dr_feature[dr_id] = feature
        return feature

    def pack_dataset(self):
        if self.split == "test":
            test_dat = open(f'./{self.output}/test.dat', 'w', encoding='utf-8')
        for (q_idx, q) in enumerate(tqdm(self.q_list, desc=f'pack {self.split} dataset')):
            q_feature = self.q_emb[q]
            pos_dr = self.q_dr_match[q]
            pos_feature = self.dr_feature[pos_dr][:]
            pos_feature.append(q_feature)
            if self.split == 'test':
                print(f'# query {q_idx+1} {q} {pos_dr}', file=test_dat)
                print(f"1 'qid':{q_idx+1} # doctor: {pos_dr}", file=test_dat)
            self.features.append(pos_feature)
            self.labels.append(1)

            # negtive sampling
            neg_drs = self.dr_list[:]
            neg_drs.remove(pos_dr)
            if self.split != 'test':
                neg_drs = sample(neg_drs, self.neg_sample)
            else:
                neg_drs = self.most_common_drs[:]
                neg_drs.remove(pos_dr)
                neg_drs = neg_drs[:100] # other top 100 doctors handling the most queries except current doctor
            for neg_dr in neg_drs:
                neg_feature = self.dr_feature[neg_dr][:]
                neg_feature.append(q_feature)
                if self.split == 'test':
                    print(f"0 'qid':{q_idx+1} # doctor: {neg_dr}", file=test_dat)
                self.features.append(neg_feature)
                self.labels.append(0)
        if self.split == 'test':
            test_dat.close()

class NLIDataset(Dataset):
    def __init__(self, split, tokenizer):
        self.split = split
        self.df = pd.read_csv(f'./dataset/{split}.csv', sep='\t', encoding='utf-8')
        self.len = len(self.df)
        self.tokenizer = tokenizer
    
    def __getitem__(self, idx):
        text_a, text_b, label = self.df.iloc[idx, :].values
        label_id = int(label)
        label_tensor = torch.tensor(label_id)
        word_pieces = ['[CLS]']
        
        tokens_a = self.tokenizer.tokenize(str(text_a))
        if len(tokens_a) >= 250:
            tokens_a = tokens_a[:250]
        word_pieces += tokens_a + ['[SEP]']
        len_a = len(word_pieces)
        
        tokens_b = self.tokenizer.tokenize(str(text_b))
        if len(tokens_b) >= 250:
            tokens_b = tokens_b[:250]
        word_pieces += tokens_b + ['[SEP]']
        len_b = len(word_pieces) - len_a
        
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        segments_tensor = torch.tensor([0] * len_a + [1] * len_b, dtype=torch.long)
        
        return tokens_tensor, segments_tensor, label_tensor
    
    def __len__(self):
        return self.len