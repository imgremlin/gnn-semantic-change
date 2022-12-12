import argparse
import gzip
import os
import pickle
import re
from collections import Counter
from itertools import chain, combinations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm import tqdm


def read_file() -> list:
    "Read file and basic preprocessing"
    with gzip.open(args.data_path + f'ccoha{args.file_id}.txt.gz',
                   mode="rb") as f:
        file_content = f.read()
    file_content = file_content.decode()
    fclean = file_content.split('\n')
    return fclean

def clean_text(txt: str) -> str:
    """
    Clean text from special characters,
    punctuation and double-spaces
    """
    txt = re.sub(r'(\S*\d\S*)', 'NUM', txt)
    txt = re.sub(r"(\s*\w*[<'>]\w*\s*)", ' ',txt)
    txt = re.sub(r"[-.#*&/%*=@$;,]", ' ',txt)
    txt = re.sub(r"[\[\]\(\)]", ' ',txt)
    txt = re.sub(r"\s{2,}"," ", txt)
    txt = re.sub(r"^\s","", txt)
    txt = re.sub(r"\s$","", txt)
    return txt

def chunks(l: list, n: int):
    "Yield successive n-sized chunks from l"
    for i in range(0, len(l), n):
        yield l[i:i + n]
        
def compute_coocurences(fclean: list):
    """
    Compute coocurence for each pair of words within a context window
    """
    num_batches = len(fclean)//args.batch_size + int(len(fclean)%args.batch_size!=0)
    print('number of batches:', num_batches)
    
    # calculate C^2_N coocurences,
    # where N - window_size
    big_sum = Counter([])
    for batch in tqdm(chunks(fclean, args.batch_size), total=num_batches):
        tdf=[]
        for sent in batch:
            sent = clean_text(sent)
            st = sent.split()
            for i in range(len(st)-args.window_size+1):
                tdf.append(list(combinations(st[i:args.window_size+i], 2)))
        big_sum += Counter(list(chain(*tdf)))
    
    pairs = big_sum.most_common()
    values_arr, word_arr, context_arr = [], [], []
    for p in pairs:
        values_arr.append(p[1])
        word_arr.append(p[0][0])
        context_arr.append(p[0][1])
        
    # check the alphabetic order to avoid repetitions
    arr_1, arr_2 = [],[]
    for i in range(len(word_arr)):
        if word_arr[i] >= context_arr[i]:
            arr_1.append(word_arr[i])
            arr_2.append(context_arr[i])
        else:
            arr_2.append(word_arr[i])
            arr_1.append(context_arr[i])
            
    return arr_1, arr_2, values_arr

def compute_attributes(
    arr_1: list, arr_2: list,
    values_arr: list
    ) -> pd.DataFrame:
    """
    Save dataframe with attributes and 
    mapping between products and their IDs
    """
    df = pd.DataFrame(
        zip(arr_1, arr_2, values_arr),
        columns=['w_a','w_b','val']
    )
    
    # delete self-loops
    df = df[df['w_a'] != df['w_b']]
    
    df = df.groupby(['w_a','w_b'])['val'].sum().reset_index()
    solo_col = pd.concat(
        [
            df[['w_a','val']].rename(columns={'w_a':'w'}),
            df[['w_b','val']].rename(columns={'w_b':'w'}),
        ], axis=0
    )
    single_sum = solo_col.groupby('w')['val'].sum().to_dict()

    total_words = df['val'].sum()
    print('total number of words:', total_words)
    
    # calculate p(a), p(b), p(a^b) for PPMI
    df['p_a'] = df['w_a'].apply(lambda x: single_sum[x] / total_words)
    df['p_b'] = df['w_b'].apply(lambda x: single_sum[x] / total_words)
    df['p_a_b'] = df['val'] / total_words
    
    # crop by minimum num of words
    df = df[df['val'] >= args.min_words].reset_index(drop=True)
    
    # PPMI (LINK TARGET)
    df['pmi'] = np.log2(df['p_a_b'] / (df['p_a'] * df['p_b']))
    df['ppmi'] = (df['pmi'] - args.alpha).apply(lambda x: max(0,x))
    
    # SAMPLE_WEIGHT (LINK WEIGHT)
    df['sample_weight'] = np.log2(df['val'])
    df['sample_weight'] /= df['sample_weight'].mean()
    
    return df

def dump_data(df: pd.DataFrame) -> None:
    """
    Save dataframe with attributes and 
    mapping between products and their IDs
    """
    os.makedirs(args.save_path, exist_ok = True)
    DF_PATH = args.save_path + f'df_{args.file_id}_{args.window_size}.feather'
    DICT_PATH = args.save_path + f'unique_products_{args.file_id}_{args.window_size}.pkl'
    df.to_feather(DF_PATH)
    
    print("dataframe with attributes saved to:", DF_PATH)
    
    unique_prods = list(set(df['w_a']).union(set(df['w_b'])))
    mapping = dict(zip(unique_prods, range(len(unique_prods))))
    with open(DICT_PATH, 'wb') as f:
        pickle.dump(mapping, f)
        
    print("product dict saved to:", DICT_PATH)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_id', type=int, default=1,
                        help="which corpus to preprocess (1 or 2")
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--data_path', type=str, default='data/init/')
    parser.add_argument('--save_path', type=str, default='data/prepro/')
    parser.add_argument('--alpha', type=int, default=4,
                        help='alpha value in PPMI')
    parser.add_argument('--min_words', type=int, default=5,
                        help='break point or not')
    parser.add_argument('--window_size', type=int, default=5,
                        help='context window size')
  
    args = parser.parse_args()
    
    fclean = read_file()
    arr_1, arr_2, values_arr = compute_coocurences(fclean)
    df = compute_attributes(arr_1, arr_2, values_arr)
    
    dump_data(df)
