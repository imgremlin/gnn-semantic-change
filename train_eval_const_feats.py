import argparse
import pickle
from itertools import chain

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding, Linear
from torch_geometric.data import Data
from torch_geometric.nn import ARMAConv
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def prepro():
    "Read data and convert it to graph format"
    df = pd.read_feather(
        args.data_path + f'df_{args.file_id}_{args.window_size}.feather')
    with open(
        args.data_path + f'unique_products_{args.file_id}_{args.window_size}.pkl',
        'rb') as f:
        unique_products = pickle.load(f)
    
    df['w_a_map'] = df['w_a'].apply(lambda x: unique_products[x])
    df['w_b_map'] = df['w_b'].apply(lambda x: unique_products[x])
    
    node_features = torch.ones(len(unique_products)).reshape(-1,1)
    edges = torch.tensor(
        df[['w_a_map','w_b_map']].to_numpy().transpose(),
        dtype=torch.long
    )
    sample_weight = torch.tensor(
        df['sample_weight'].to_numpy(),
        dtype=torch.float32
    )
    edge_target = torch.tensor(
        df['ppmi'].to_numpy(),
        dtype=torch.float32
    )
    
    data = Data(
        x = node_features,
        edge_index = edges,
        y = edge_target,
        edge_weight = sample_weight
    )
    
    
    return {'df': df, 'unique_products': unique_products, 'graph': data}

def chunks(l: list, n: int):
    "Yield successive n-sized chunks from l"
    for i in range(0, len(l), n):
        yield l[i:i + n]

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_lin_1 = Linear(1, 64)
        self.gcn = ARMAConv(64,32)
        self.gcn2 = ARMAConv(32,32)
        self.post_lin = Linear(32,16)
        self.decode_lin2 = Linear(32,1)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        
    def encode(self, x, edge_index, edge_weight):
        x = F.elu(self.pre_lin_1(x))
        x = F.elu(self.gcn(x, edge_index, edge_weight))
        x = F.elu(self.gcn2(x, edge_index, edge_weight))
        x = self.post_lin(x)
        return x
    
    def decode(self, z, edge_label_index):
        return self.cos(z[edge_label_index[0]], z[edge_label_index[1]])
    
def train(model, optimizer, criterion, data, mask_split, treshold = 1e-5):
    "Training pipeline with asymmetric transductive validation"
    model.train()
    optimizer.zero_grad()
    
    pos_mask = data.y >= treshold
    pos_train_idx = ((pos_mask*mask_split) == True).nonzero(as_tuple=True)[0]
    pos_test_idx = ((pos_mask*(~mask_split)) == True).nonzero(as_tuple=True)[0]
    
    neg_train_idx = (
        ((~pos_mask)*mask_split)==True
    ).nonzero(as_tuple=True)[0]
    perm = torch.randperm(neg_train_idx.size(0))
    sampled_neg_idx = perm[:pos_mask.sum(0)]
    
    edge_index_encode = torch.cat([
        data.edge_index[:,pos_train_idx],
        data.edge_index[:,pos_test_idx]
        ], dim =- 1,
        )
    edge_label_encode = torch.cat([
            data.y[pos_train_idx],
            data.y[pos_test_idx],
        ], dim=0
        )
    edge_weight_encode = torch.cat([
            data.edge_weight[pos_train_idx],
            data.edge_weight[pos_test_idx],
        ], dim=0
        )
    
    edge_index_decode = torch.cat([
        data.edge_index[:,pos_train_idx],
        data.edge_index[:,pos_test_idx],
        data.edge_index[:,sampled_neg_idx]
        ], dim=-1,
        )
    edge_label_decode = torch.cat([
            data.y[pos_train_idx],
            data.y[pos_test_idx],
            data.y[sampled_neg_idx],
        ], dim=0
        )
    
    z = model.encode(
        data.x,
        edge_index_encode,
        edge_weight_encode
        )
    out = model.decode(z, edge_index_decode).view(-1)
    
    loss_mask = torch.tensor(
        [True] * pos_train_idx.size(0) + 
        [False] * pos_test_idx.size(0) +
        [True] * sampled_neg_idx.size(0)
    ).to(args.device)
    
    loss = criterion(
        torch.masked_select(out, loss_mask),
        torch.masked_select(edge_label_decode, loss_mask)).mean()
    loss.backward()
    optimizer.step()
    
    return z, loss

@torch.no_grad()
def test(model, criterion, data, mask_split, treshold = 1e-5):
    "Collect predictions for 'positive' and 'negative' edges"
    model.eval()
    pos_mask = data.y >= treshold
    pos_idx = ((pos_mask)==True).nonzero(as_tuple=True)[0]
    pos_test_idx = ((pos_mask*mask_split) == True).nonzero(as_tuple=True)[0]
    neg_test_idx = (
        (~pos_mask*mask_split) == True
    ).nonzero(as_tuple=True)[0]
    
    z = model.encode(
        data.x,
        data.edge_index[:,pos_idx],
        data.edge_weight[pos_idx]
        )
    
    pos_out = model.decode(z, data.edge_index[:,pos_test_idx]).view(-1)
    neg_out = model.decode(z, data.edge_index[:,neg_test_idx]).view(-1)
    
    loss_pos = criterion(pos_out, data.y[pos_test_idx]).mean()
    loss_neg = criterion(neg_out, data.y[neg_test_idx]).mean()
    
    return loss_pos, loss_neg

def create_mask(data):
    "Train/test mask for transductive validation"
    num_edges = data.edge_index.shape[1]
    num_train_edges = int(num_edges*0.7)
    num_test_edges = num_edges-num_train_edges
    mask_bool = torch.tensor(
        [True]*num_train_edges+[False]*num_test_edges,
        dtype=torch.bool
    )
    idx = torch.randperm(mask_bool.shape[0])
    mask_split = mask_bool[idx].view(mask_bool.size())
    
    return mask_split

def train_pipe(pipe):
    "Train/test loop"
    data = pipe['graph'].to(args.device)
    model = Net().to(args.device)
    mask = pipe['mask'].to(args.device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.L1Loss(reduction='none')
    train_arr, val_pos, val_neg, val_arr = [], [], [], []
    
    for epoch in range(1, args.epochs + 1):
        z, loss = train(model, optimizer, criterion, data, mask)
        pos, neg = test(model, criterion, data, ~mask)
        pos_val = pos.cpu().detach().cpu().numpy()
        neg_val = neg.cpu().detach().numpy()
        val_loss = pos_val + neg_val
        val_pos.append(pos_val)
        val_neg.append(neg_val)
        val_arr.append(val_loss)
        train_arr.append(loss.cpu().detach().numpy())
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f} Val:{val_loss:.4f} Pos: {pos:.4f} Neg: {neg:.4f}")
    
    pipe['model'] = model
    pipe['loss_plots'] = {'train_arr':[], 'val_pos':[], 'val_neg':[],}
    pipe['loss_plots']['train_arr'] = train_arr
    pipe['loss_plots']['val_arr'] = val_arr
    pipe['loss_plots']['val_pos'] = val_pos
    pipe['loss_plots']['val_neg'] = val_neg
    
    if args.save_model:
        torch.save(pipe_dict['model'].state_dict(), args.save_model_path)
    
    return pipe

def predict_all(pipe):
    "Collect all word embeddings"
    data = pipe['graph'].to(args.device)
    model = pipe['model'].to(args.device)
    with torch.no_grad():
        model.eval()
        pipe['preds'] = model.encode(data.x, data.edge_index, data.edge_weight).cpu().numpy()
    return pipe

def dump_data():
    del pipe_dict['model']
    del pipe_dict['graph']
    with open(args.data_path+f'pipe_{args.file_id}.pkl', 'wb') as f:
        pickle.dump(pipe_dict, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_id', type=int, default=1,
                        help="on which corpus to train the model (1 or 2")
    parser.add_argument('--window_size', type=int, default=5,
                        help='place to save or load the model')
    
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-4)

    parser.add_argument('--data_path', type=str, default='data/prepro/')
    parser.add_argument('--save_model', type=bool, default=False)

    parser.add_argument('--load_model', type=bool, default=False,
                        help='use model on inference or train from scratch')
    parser.add_argument('--load_model_path', type=str, default='data/pretrained_model.pt')
    parser.add_argument('--save_model_path', type=str, default='data/saved_model.pt')

  
    args = parser.parse_args()
    
    pipe_dict = {}
    pipe_dict = prepro()

    pipe_dict['mask'] = create_mask(pipe_dict['graph'])

    if args.load_model:
        pipe_dict['model'] = Net(pipe_dict['embedding_size'])
        pipe_dict['model'].load_state_dict(torch.load(args.load_model_path))
    else:
        pipe_dict = train_pipe(pipe_dict)

    pipe_dict = predict_all(pipe_dict)

    dump_data()