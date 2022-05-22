import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
# from sklearn.feature_extraction.text import TfidfVectorizer

parser = argparse.ArgumentParser()
parser.add_argument('-seed', default=2021, type=int)
parser.add_argument('-model', default='bert', type=str)
parser.add_argument('-load_sl_model', default=1, type=int)
args = parser.parse_args()

# set you available gpus
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
np.random.seed(args.seed)
if args.model == 'bert':
    torch.manual_seed(args.seed)
    n_gpu = torch.cuda.device_count()
    print('gpu num: ', n_gpu)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def bert_sent_embed(model, tokenizer, device, id_content, output_path):
    con_emb_dict = {}
    for idx, content in tqdm(id_content.items(), desc=output_path):
        input_ids = torch.tensor([tokenizer.encode(str(content))])
        if len(input_ids[0].numpy().tolist()) > 512:
            input_ids = torch.from_numpy(np.array(input_ids[0].numpy().tolist()[0:512])).reshape(1, -1).type(torch.LongTensor)
        input_ids = input_ids.to(device)
        with torch.no_grad():
            features = model(input_ids)
        con_emb_dict[idx] = features[1].cpu().numpy()[0].tolist()
    with open(output_path, 'w') as f:
        json.dump(con_emb_dict, f, ensure_ascii=False) 
        
# def bert_dialog_turn_embed(model, tokenizer, device, dialogs, output_path):
#     turns_emb = {}
#     for idx, turns in tqdm(dialogs.items(), desc = output_path):
#         turns_emb[idx] = []
#         for turn in turns:
#             input_ids = torch.tensor([tokenizer.encode(turn)])
#             if len(input_ids[0].numpy().tolist()) > 512:
#                 input_ids = torch.from_numpy(np.array(input_ids[0].numpy().tolist()[0:512])).reshape(1, -1).type(torch.LongTensor)
#             input_ids = input_ids.to(device)
#             with torch.no_grad():
#                 features = model(input_ids) 
#             turns_emb[idx].append(features[1].cpu().numpy()[0].tolist())
#     with open(output_path, 'w') as f:
#         f.write(json.dumps(turns_emb, ensure_ascii=False)) 

# def tfidf_embed(id_texts, output_path):
#     with open('./stopwords.txt', 'r', encoding='utf-8') as f:       
#         stopwords = [line.strip() for line in f.readlines()]
#     splited_words_texts = []
#     for text in tqdm(id_texts.values(), desc='spliting words'):
#         words = [word for word in list(jieba.cut(text)) if word not in stopwords]
#         splited_words_texts.append(re.sub(r'[0-9]', '', ' '.join(words)))
#     vectorizer = TfidfVectorizer(min_df=10, max_df=150)
#     vectors = vectorizer.fit_transform(splited_words_texts) 
#     feature_names = vectorizer.get_feature_names()
#     dense = vectors.todense()
#     denselist = dense.tolist()
#     df = pd.DataFrame(denselist, columns=feature_names)

#     con_emb_dict = {}
#     for idx, id_text in enumerate(tqdm(id_texts.items())):
#         text_id, _ = id_text
#         embedding = df.loc[idx].tolist()
#         con_emb_dict[text_id] = embedding
#     with open(output_path, 'w', encoding = 'utf-8') as f:
#         json.dump(con_emb_dict, f, ensure_ascii=False)
        
def chunks(list, n):
    chunks_list = []
    len_list = len(list)
    step = math.ceil(len_list / n)
    for i in range(0, n):
        chunks_list.append(list[i*step:(i+1)*step])
    return chunks_list

def main():
    df = pd.read_csv(f'./dataset/embed.csv', delimiter='\t', encoding='utf-8')
    df = df[['dr_id', 'dialog_id', 'q', 'parsed_dialog']]
    id_profile = {}
    with open(f'./dataset/dr_profile.jsonl', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = json.loads(line)
            id_profile[line['id']] = line['goodat'] # use goodat as doctor profile
    id_q = dict(zip(df.dialog_id.tolist(), df.q.tolist()))
    id_dialog = dict(zip(df.dialog_id.tolist(), df.parsed_dialog.tolist()))
    
    if args.model == 'bert':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        tokenizer = BertTokenizer.from_pretrained('./mc_bert_base/')
        model = BertModel.from_pretrained('./mc_bert_base/')
        model = model.to(device)
        if args.load_sl_model:
            model_path = './sl_best_model/sl_best_model.bin'
            print('Load model from ' + model_path)
            loaded_dict = torch.load(model_path)
            model.state_dict = loaded_dict
            embedding_path = './bert_embeddings'
        else:
            embedding_path = './bert_embeddings_wo_sl'
        if not os.path.exists(embedding_path):
            os.makedirs(embedding_path)
            
        bert_sent_embed(model, tokenizer, device, id_profile, f'{embedding_path}/profile_embeddings.json')
        bert_sent_embed(model, tokenizer, device, id_q, f'{embedding_path}/q_embeddings.json')
        bert_sent_embed(model, tokenizer, device, id_dialog, f'{embedding_path}/dialog_embeddings.json')
        # bert embed train dialogue turns in chunks with multithreading
        # with open(f'./dataset/dialogs.json', 'r', encoding='utf-8') as f:
        #     dialogs = json.load(f)
        # train_df = pd.read_csv(f'./dataset/train.csv', delimiter='\t', encoding='utf-8')
        # train_dialog_ids = train_df.dialog_id.tolist()
        # n = 20
        # chunks_list = chunks(train_dialog_ids, n)
        # threads_list = []
        # for index in range(0, n):
        #     chunk_dialogs = {dialog_id: dialogs[dialog_id] for dialog_id in chunks_list[index]}
        #     thread = threading.Thread(
        #         target=bert_dialog_turn_embed, 
        #         args=(model, tokenizer, device, index, chunk_dialogs, f'{embedding_path}/train_turns_emb{index}.json'))
        #     threads_list.append(thread) 
        # for t in threads_list:
        #     t.setDaemon(True)
        #     t.start()
        # for t in threads_list:
        #     t.join()
        
    # if args.model == 'tfidf':
    #     if not os.path.exists('./tfidf_embeddings'):
    #         os.makedirs(f'./tfidf_embeddings')
    #     tfidf_embed(id_profile, 'tfidf_embeddings/profile_embeddings.json')
    #     tfidf_embed(id_q, './tfidf_embeddings/q_embeddings.json')
    #     tfidf_embed(id_dialog, './tfidf_embeddings/dialog_embeddings.json')

if __name__ == '__main__':
    main()
