import argparse

def init_opts(parser: argparse.ArgumentParser):
    parser.add_argument('-seed', type=int)
    parser.add_argument('-gpu', type=int)
    parser.add_argument('-name', type=str)
    parser.add_argument('-embeddings_path', type=str)
    parser.add_argument('-output_dir', type=str)

def train_opts(parser: argparse.ArgumentParser):
    parser.add_argument('-epoch_num', default=50, type=int)
    parser.add_argument('-batch_size', default=256, type=int)
    parser.add_argument('-patience', default=7, type=int)
    parser.add_argument('-lr', default=0.008, type=float)
    parser.add_argument('-dropout', default=0.2, type=float)
    parser.add_argument('-in_size', default=768, type=int)
    parser.add_argument('-hidden_size', default=256, type=int)
    parser.add_argument('-dr_dialog_sample', default=100, type=int)
    parser.add_argument('-neg_sample', default=10, type=int)
    
def eval_opts(parser: argparse.ArgumentParser):
    parser.add_argument('-eval_model', default="best_model.pt", type=str)

def multihead_att_opts(parser: argparse.ArgumentParser):
    parser.add_argument('-head_num', default=6, type=int)
    parser.add_argument('-add_self_att_on', default="none", type=str)    
