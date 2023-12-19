from __future__ import annotations

import json
import os
import random
from typing import Any


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm
from utils.data_gen_pretrained_aligned import data_gen
import argparse



def load_text_sequence_data(seq_length, scale, preprocess_text):
    mat, target , target_subj, mask, words, label_dict, label_arr = data_gen(seq_length = seq_length, scale = scale, preprocess_text= preprocess_text)
    return mat, target , target_subj, mask, words, label_dict, label_arr


def write_npys(
    label_arr: np.ndarray,
    data_arr_CNNs: list[np.ndarray],
    label_dict: dict[str, Any],
    split_criterion: str,
    suffixe: list[str],
    save_path: str = '',
) -> int:
    with open(f'{save_path}labels_dict.json', 'w') as fp:
        json.dump(label_dict, fp)

    split_criterion_dict = {
        'subj': 0,
        'book': 1,
    }
    if split_criterion != 'book-page':
        splitkeys = np.array(
            sorted(
                list(set(label_arr[:, split_criterion_dict[split_criterion]])),
            ),
        )
        if split_criterion == 'book':
            n_folds = 4
        else:
            n_folds = 5
    else:
        splitkeys = data_arr_CNNs[0]
        n_folds = 5

    outer_cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(splitkeys)):

        random.seed(fold)
        np.random.seed(fold)

        print(f'Writing fold {fold}...')

        for idx in range(len(data_arr_CNNs)):
            data_arr_CNN = data_arr_CNNs[idx]
            suffix = suffixe[idx]
            if isinstance(data_arr_CNN, pd.DataFrame):
                data_arr_CNN = data_arr_CNN.values
            if split_criterion != 'book-page':
                N_train_sub = splitkeys[train_idx]
                N_test_sub = splitkeys[test_idx]
                print(f'training split: {N_train_sub}')
                print(f'test split: {N_test_sub}')

                X_train_CNN = data_arr_CNN[
                    np.isin(
                        label_arr[:, split_criterion_dict[split_criterion]],
                        N_train_sub,
                    )
                ]
                with open(f'{save_path}/{split_criterion}/X_train_{split_criterion}_{fold}{suffix}.npy', 'wb') as f:  # noqa: E501
                    np.save(f, X_train_CNN)
                X_test_CNN = data_arr_CNN[
                    np.isin(
                        label_arr[:, split_criterion_dict[split_criterion]],
                        N_test_sub,
                    )
                ]
                with open(f'{save_path}/{split_criterion}/X_test_{split_criterion}_{fold}{suffix}.npy', 'wb') as f:  # noqa: E501
                    np.save(f, X_test_CNN)
                y_train_CNN = label_arr[
                    np.isin(
                        label_arr[:, split_criterion_dict[split_criterion]],
                        N_train_sub,
                    )
                ]
                with open(f'{save_path}/{split_criterion}/y_train_{split_criterion}_{fold}{suffix}.npy', 'wb') as f:  # noqa: E501
                    np.save(f, y_train_CNN)
                y_test_CNN = label_arr[
                    np.isin(
                        label_arr[:, split_criterion_dict[split_criterion]],
                        N_test_sub,
                    )
                ]
                with open(f'{save_path}/{split_criterion}/y_test_{split_criterion}_{fold}{suffix}.npy', 'wb') as f:  # noqa: E501
                    np.save(f, y_test_CNN)
            else:
                print("Writing for book-page")
                X_train_CNN = data_arr_CNN[train_idx]
                with open(f'{save_path}/{split_criterion}/X_train_{split_criterion}_{fold}{suffix}.npy', 'wb') as f:  # noqa: E501
                    np.save(f, X_train_CNN)
                X_test_CNN = data_arr_CNN[test_idx]
                with open(f'{save_path}/{split_criterion}/X_test_{split_criterion}_{fold}{suffix}.npy', 'wb') as f:  # noqa: E501
                    np.save(f, X_test_CNN)
                y_train_CNN = label_arr[train_idx]
                with open(f'{save_path}/{split_criterion}/y_train_{split_criterion}_{fold}{suffix}.npy', 'wb') as f:  # noqa: E501
                    np.save(f, y_train_CNN)
                y_test_CNN = label_arr[test_idx]
                with open(f'{save_path}/{split_criterion}/y_test_{split_criterion}_{fold}{suffix}.npy', 'wb') as f:  # noqa: E501
                    np.save(f, y_test_CNN)
    return 0


def main() -> int:
    # label_arr, data_arr_CNN, label_dict, fix_data = load_text_sequence_data()
    parser = argparse.ArgumentParser()
    parser.add_argument('-seq_len', '--seq_len', type=int, default=50)
    parser.add_argument('-scale', '--scale', type=bool, default=True)
    parser.add_argument('-preprocess_text', '--preprocess_text', type=bool, default=True)

    #preprocess_text
    
    args = parser.parse_args()
    
    mat, target , target_subj, mask, words, label_dict, label_arr = load_text_sequence_data(seq_length = args.seq_len, scale = args.scale, preprocess_text = args.preprocess_text)
    print("mat shape", mat.shape)
    print("label_arr shape,", label_arr.shape)
    print("target shape", len(target))
    print("target_subj shape", len(target_subj))
    print("mask shape", len(mask))
    print("words shape",words.shape)
    
    
    save_path = 'paper_splits/'
    os.makedirs(save_path, exist_ok=True)
    for split_criterion in ['subj', 'book', 'book-page']:
        os.makedirs(os.path.join(save_path, split_criterion), exist_ok=True)
        print(f'Creating files for split {split_criterion}...')
        write_npys(
            label_arr=label_arr,
            data_arr_CNNs=[mat, words],
            label_dict=label_dict,
            split_criterion=split_criterion,
            suffixe=['', '_fix_data'],
            save_path=save_path,
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
