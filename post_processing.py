import argparse
import pickle

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def load_data():
    pipe_dict = {}
    with open(args.data_path + f'pipe_1.pkl', 'rb') as f:
        pipe_dict['1'] = pickle.load(f)

    with open(args.data_path + f'pipe_2.pkl', 'rb') as f:
        pipe_dict['2'] = pickle.load(f)
    
    targets = pd.read_csv(args.target_data, sep='\t', header=None)
    targets.columns = ['word', 'target']
    
    return pipe_dict, targets

def cossim(arr1, arr2):
    return arr1.dot(arr2) / (np.sqrt((arr1**2).sum()) * np.sqrt((arr2**2).sum()))

def eucl(arr1, arr2):
    return 1 - (np.abs(arr1 - arr2) ** 2).mean()

def compute_dist(pipe_dict):
    "Compute euclidian distance between target words"
    mat1 = pipe_dict['1']['mat']
    mat2 = pipe_dict['2']['mat']
    cosdists_arr = []
    for w in targets['word']:
        ind = pipe_dict['common_words'].index(w)
        val = eucl(mat1[ind], mat2[ind])    
        cosdists_arr.append(val)
    pipe_dict['preds'] = cosdists_arr
    
    return pipe_dict

def get_common_words(pipe_dict):
    "Find and filter common words from two corpora"
    common_words = list(
        set(pipe_dict['1']['unique_products'].keys())&set(pipe_dict['2']['unique_products'].keys())
    )

    word_freq = pd.DataFrame()
    word_freq['word'] = common_words
    word_freq['corp1'] = [pipe_dict['1']['unique_products'][w] for w in common_words]
    word_freq['corp2'] = [pipe_dict['2']['unique_products'][w] for w in common_words]
    
    thr1 = word_freq['corp1'].quantile(args.min_align_quant)
    thr2 = word_freq['corp2'].quantile(args.min_align_quant)
    word_freq = word_freq[
        (word_freq['corp1'] > thr1) & (word_freq['corp2'] > thr2)
    ]
    word_freq = word_freq.reset_index(drop=True)

    common_words = list(word_freq['word'].values) 
    common_words = common_words + list(set(targets['word']) - set(common_words))
    pipe_dict['common_words'] = common_words
    
    return pipe_dict

def get_matrices(pipe_dict):
    "Align two matrices with word embeddings"
    embs1, embs2 = [],[]
    dct1 = pipe_dict['1']['unique_products']
    dct2 = pipe_dict['2']['unique_products']
    for w in pipe_dict['common_words']:
        embs1.append(pipe_dict['1']['preds'][dct1[w]])
        embs2.append(pipe_dict['2']['preds'][dct1[w]])

    mat1 = np.array(embs1)
    mat2 = np.array(embs2)

    R, _ = orthogonal_procrustes(embs1, embs2)
    mat1 = mat1 @ R

    pipe_dict['1']['mat'] = mat1
    pipe_dict['2']['mat'] = mat2

    return pipe_dict

def get_results():
    "Print accuracy, Confusion Matrix and POS Table"
    targets['preds'] = pipe_dict['preds']
    classif_threshold = targets['preds'].quantile(args.classif_quant)
    targets['bin_preds'] = targets['preds'].apply(lambda x: x >= classif_threshold).astype(int)
    print(f"Accuracy:", accuracy_score(targets['target'], targets['bin_preds']), '\n')

    cm = 100 * confusion_matrix(targets['target'], targets['bin_preds']) / len(targets)
    cm = cm.round(1)
    print('Confusion Matrix:\n', cm, '\n')

    targets['pos'] = targets['word'].apply(lambda x: x.split('_')[-1])

    pos_counts = targets.groupby('pos')['word'].count()
    pos_counts.name = 'count'

    pos_acc = targets.groupby('pos').apply(
        lambda x: accuracy_score(x['target'], x['bin_preds'])
    )
    pos_acc.name = 'accuracy'
    
    merge_pos = pd.merge(
        pos_acc, pos_counts,
        right_index = True, left_index = True
    )
    print('POS Table:\n', merge_pos)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='data/prepro/')
    parser.add_argument('--target_data', type=str, default='data/init/binary.txt')
    parser.add_argument('--min_align_quant', type=float, default=0.15)
    parser.add_argument('--classif_quant', type=float, default=0.52)
    
    args = parser.parse_args()
    
    pipe_dict, targets = load_data()
    pipe_dict = get_common_words(pipe_dict)

    pipe_dict = get_matrices(pipe_dict)
    pipe_dict = compute_dist(pipe_dict)

    get_results()
