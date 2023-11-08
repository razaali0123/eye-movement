from __future__ import annotations

import argparse
import json
import os
import random
import sys

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import GlobalMaxPool1D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from transformers import AutoTokenizer, TFDistilBertForSequenceClassification


# from keras.models import Sequential
from keras.layers import LSTM
# from keras.layers import Dense
# from keras.layers import TimeDistributed
from keras.layers import Bidirectional

# from utils import feature_extraction as feature_extraction
sys.path.append(os.getcwd())

def whole_book_analysis(book_list, df_cognitive):
  for cnt,b in enumerate(book_list):
    check = df_cognitive[df_cognitive['page_name'] == b].sort_values(['pnr', 'word_loc']).loc[:, ['pnr', 'page_name', 'word', 'word_loc', 'acc']]
    lst = check.groupby('pnr')['word'].apply(list)
    
    check_acc = pd.DataFrame(check.groupby(['pnr', 'page_name'])['acc'].apply(list))
    check_acc = check_acc.acc.apply(lambda x: x[0])
    check = pd.DataFrame(check_acc).merge(pd.DataFrame(lst), on = ['pnr'])
    # check['unique_word_count'] = check.word.apply(lambda x: len(np.unique(x)))
    if cnt == 0:
      final_check = check.copy()
    else:
      final_check = pd.concat([final_check, check])

  return final_check






def get_nn_model(dropout, x_train, input_shape):
    # df_cognitive = pd.read_csv("/home/raza/repo/etra-reading-comprehension/SB-SAT/fixation/df_cognitive.csv")
    
    # distill_model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", output_hidden_states = True)


    dropout_rate = dropout

    # input_shape = 100
    lstm_input = tf.keras.Input(shape=( input_shape,),dtype='float32')
    embeddings = tf.keras.layers.Embedding(1000, 64, input_length=input_shape)(lstm_input)

    # transformer_input = tf.keras.Input(shape=(input_shape, 768),dtype='float32')
    # # att = tf.keras.Input(shape=(input_shape,),dtype='int32')

    # lstm_input = tf.keras.Input(shape=(input_shape, x_train.shape[2]),dtype='float32')

    # lstm = Bidirectional(LSTM(16, dropout=0.2)(lstm)
    # lstm = Bidirectional(LSTM(3, dropout=0.2)(lstm)



    # output = distill_model([ids,att])
    # output = output.hidden_states
    # n = len(output)
    # lst = [output[o] for o in range(n)]
    # concat_transformer = tf.keras.layers.concatenate(lst, axis  = 2, name = 'concat_transformer')
    # concat = tf.keras.layers.concatenate([transformer_input, lstm_input], axis  = 2, name = 'concat')
    # # output = tf.keras.layers.Flatten()(output)
    # ## lstm addition
    # # lstm = Bidirectional(LSTM(256, dropout=0.1, return_sequences=True))(lstm_input)
    # lstm = Bidirectional(LSTM(256, dropout=dropout_rate,  return_sequences=True))(concat)
    # lstm = Bidirectional(LSTM(128, dropout=dropout_rate))(lstm)

    # output_lstm = tf.keras.layers.Dense(64,activation='relu')(lstm)
    # output_lstm = tf.keras.layers.Dense(3,activation='softmax')(lstm)



    ## Concat
    # concat = tf.keras.layers.concatenate([output, lstm], axis  = 1, name = 'concat')

    ## Now dense continues

    output = tf.keras.layers.Dense(128,activation='relu')(embeddings)

    output = tf.keras.layers.Dropout(dropout_rate)(output)

    output = tf.keras.layers.Dense(64,activation='relu')(output)

    output = tf.keras.layers.Dropout(dropout_rate)(output)

    output = tf.keras.layers.Dense(32,activation='relu')(output)

    output = tf.keras.layers.Dropout(dropout_rate)(output)

    output = tf.keras.layers.Dense(1 ,activation='sigmoid')(output)
    model = tf.keras.models.Model(inputs = lstm_input,outputs = output)
    # model.compile(Adam(lr=6e-6), loss='binary_crossentropy', metrics=['accuracy'])
    # for i in model.layers:
    #     if (i.name.startswith('tf_distil')):
    #         i.trainable = False
    #     else:
    #         i.trainable = True
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), metrics= ['AUC', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        
    return model


def help_roc_auc(y_true, y_pred):
    if len(np.unique(y_true)) == 1:
        return .5
    else:
        return roc_auc_score(y_true, y_pred)

# calculate the roc-auc as a metric


def auroc(y_true, y_pred):
    return tf.py_function(help_roc_auc, (y_true, y_pred), tf.double)

def transformer_encode(data,maximum_length, tokenizer) :
  input_ids = []
  attention_masks = []
  

  for i in range(len(data.text)):
      encoded = tokenizer.encode_plus(
        
        data.text[i],
        add_special_tokens=True,
        pad_to_max_length=True,
        truncation = True,
        max_length=maximum_length,
        
        return_attention_mask=True,
        
      )
      
      input_ids.append(encoded['input_ids'])
      attention_masks.append(encoded['attention_mask'])
  return np.array(input_ids), np.array(attention_masks)


def train_nn(
    spit_criterions, labels,
    feature_names_per_word,
    model_name,
    input_shape,
    dropout,
    tuning,
    seq_list,
    dropout_list,
    # flag_sequence_bilstm=True,
    # word_in_fixation_order=True,
    # use_reduced_pos_sequence=True,
    # use_content_word_sequence=True,
    # use_numeric=True,
    # use_fixation_sequence=True,
    flag_redo=False,
    normalize_flag=True,
    use_gaze_entropy_features=True,
    patience=50,
    batch_size=256,
    epochs=1000,
    save_dir='/home/raza/repo/etra-reading-comprehension/nn/results/',
    save_csv=True,
    save_joblib=False,
):
    # tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    final_df = {}
    final_df['seq_len'] = []
    final_df['dropout'] = []
    final_df['label'] = []
    final_df['split_criterion'] = []
    
    for split_criterion in spit_criterions:
        for label in labels:

            # model_prefix = str(flag_sequence_bilstm) +\
            #     '_' + str(word_in_fixation_order) +\
            #     '_' + str(use_reduced_pos_sequence) +\
            #     '_' + str(use_content_word_sequence) +\
            #     '_' + str(use_numeric) +\
            #     '_' + str(use_fixation_sequence) +\
            #     '_'
            model_prefix = "_new_prefix_"
            csv_save_path = f'{save_dir}{model_prefix}_dropout_{dropout}_seqlen_{input_shape}_{split_criterion}_text_sequence_{label}.csv'  # noqa: E501
            joblib_save_path = csv_save_path.replace('.csv', '.joblib')
            if not flag_redo and save_csv and os.path.exists(csv_save_path):
                continue
            if not flag_redo and save_joblib and os.path.exists(joblib_save_path):
                continue
            SB_SAT_PATH = f'/content/paper_splits/{split_criterion}/'
            split_criterion_dict = {
                'subj': 0,
                'book': 1,
                'subj-book': 0,
            }
            with open('/content/paper_splits/labels_dict.json') as fp:
                label_dict = json.load(fp)

            if (split_criterion == 'book') or (split_criterion == 'subj-book'):
                num_folds = 4
            else:
                num_folds = 5
            pd_init = pd.DataFrame(
                columns=[
                    'ahn_baseline',
                    'fold0_auc', 'fold1_auc', 'fold2_auc', 'fold3_auc', 'fold4_auc',
                    'fold0_tpr', 'fold1_tpr', 'fold2_tpr', 'fold3_tpr', 'fold4_tpr',
                    'fold0_fpr', 'fold1_fpr', 'fold2_fpr', 'fold3_fpr', 'fold4_fpr',
                    'fold0_y_pred', 'fold1_y_pred', 'fold2_y_pred', 'fold3_y_pred', 'fold4_y_pred',
                    'fold0_y_test', 'fold1_y_test', 'fold2_y_test', 'fold3_y_test', 'fold4_y_test',
                    'avg_auc', 'std_auc',
                ],
            )
            out_dict = dict()

            pd_init['ahn_baseline'] = [model_name]



            if tuning != True:

                for fold in range(num_folds):
                    np.random.seed(fold)
                    random.seed(fold)
                    # collect the inputs for train, validation and test
                    # use only features where flag is True
                    train_inputs = []
                    val_inputs = []
                    test_inputs = []
                    X_train_path = os.path.join(
                        SB_SAT_PATH, f'X_train_{split_criterion}_{fold}.npy',
                    )
                    X_train_fix_path = os.path.join(
                        SB_SAT_PATH, f'X_train_{split_criterion}_{fold}_fix_data.npy',
                    )
                    y_train_path = os.path.join(
                        SB_SAT_PATH, f'y_train_{split_criterion}_{fold}.npy',
                    )
                    x_train_all, y_train_all = np.load(X_train_path).astype(float), np.load(
                        y_train_path, allow_pickle=True,
                    )
                    x_train_fix_all = np.load(X_train_fix_path, allow_pickle= True)
                    x_train_fix_all = x_train_fix_all.astype("float")
                    # x_train_fix_all = tf.cast(x_train_fix_all, tf.float32)
                    
                    # x_train_fix_all = pd.DataFrame(x_train_fix_all, columns = ['text_list', 'text'])

                    
                    
                    # x_train_fix_postions = x_train_fix_all[:, :, 4]
                    if normalize_flag:
                        scaler = MinMaxScaler()
                        fix_scaler = MinMaxScaler()
                        x_train_all = scaler.fit_transform(
                            x_train_all.reshape(-1, x_train_all.shape[-1]),
                        ).reshape(x_train_all.shape)
                        x_train_fix_all = fix_scaler.fit_transform(
                            x_train_fix_all.reshape(-1, x_train_fix_all.shape[-1]),
                        ).reshape(x_train_fix_all.shape)
                        x_train_fix_all = np.where(
                            np.isnan(x_train_fix_all), -4, x_train_fix_all,
                        )
                    if split_criterion != 'book':
                        outer_cv = KFold(
                            n_splits=4, shuffle=True,
                            random_state=fold,
                        )
                    else:
                        outer_cv = KFold(
                            n_splits=3, shuffle=True,
                            random_state=fold,
                        )

                    if split_criterion != 'book-page':
                        splitkeys = np.array(
                            sorted(
                                list(
                                    set(
                                        y_train_all[
                                            :,
                                            split_criterion_dict[split_criterion],
                                        ],
                                    ),
                                ),
                            ),
                        )
                    else:
                        splitkeys = y_train_all[:, label_dict[label]]

                    for train_idx, val_idx in outer_cv.split(splitkeys):
                        break

                    if split_criterion != 'book-page':
                        N_train_sub = splitkeys[train_idx]
                        N_test_sub = splitkeys[val_idx]

                        train_idx = np.where(
                            np.isin(
                                y_train_all[
                                    :, split_criterion_dict[split_criterion],
                                ], N_train_sub,
                            ),
                        )[0]
                        val_idx = np.where(
                            np.isin(
                                y_train_all[
                                    :, split_criterion_dict[split_criterion],
                                ], N_test_sub,
                            ),
                        )[0]
                    x_train = x_train_all[train_idx]
                    y_train = y_train_all[train_idx]
                    x_val = x_train_all[val_idx]
                    y_val = y_train_all[val_idx]

                    
                    xtr_words = x_train_fix_all[train_idx, :, :]
                    val_words = x_train_fix_all[val_idx, :, :]
                    # y_train_all[val_idx]

                    y_train = np.array(y_train[:, label_dict[label]], dtype=int)
                    y_val = np.array(y_val[:, label_dict[label]], dtype=int)
                    # input_ids, attention_masks = transformer_encode(xtr_words.reset_index(drop = True), input_shape, tokenizer)
                    # val_input_ids, val_attention_masks = transformer_encode(val_words.reset_index(drop = True), input_shape, tokenizer)
                    
                    
                    train_inputs.append(xtr_words)
                    train_inputs.append(x_train)

                    
                    # train_inputs.append(input_ids)
                    # train_inputs.append(attention_masks)
                    # train_inputs.append(x_train)
                    

                    val_inputs.append(val_words)
                    val_inputs.append(x_val)
                    # val_inputs.append(val_input_ids)
                    # val_inputs.append(val_attention_masks)
                    # val_inputs.append(x_val)


                    # print("test val is ", x_val.shape)

                    # Test Data
                    X_test_path = os.path.join(
                        SB_SAT_PATH,
                        f'X_test_{split_criterion}_{fold}.npy',
                    )
                    X_test_fix_path = os.path.join(
                        SB_SAT_PATH,
                        f'X_test_{split_criterion}_{fold}_fix_data.npy',
                    )
                    y_test_path = os.path.join(
                        SB_SAT_PATH,
                        f'y_test_{split_criterion}_{fold}.npy',
                    )
                    x_test_all, y_test_all = np.load(X_test_path).astype(float), np.load(
                        y_test_path, allow_pickle=True,
                    )
                    x_test_fix_all = np.load(X_test_fix_path, allow_pickle=True).astype(float)
                    x_test_fix_all = tf.cast(x_test_fix_all, tf.float32)
                    # x_test_fix_postions = x_test_fix_all[:, :, 4]
                    # x_test_fix_all = pd.DataFrame(x_test_fix_all, columns = ['text_list', 'text'])
                    # test_input_ids, test_attention_masks = transformer_encode(x_test_fix_all.reset_index(drop = True), input_shape, tokenizer)
                    # test_inputs.append(test_input_ids)
                    # test_inputs.append(test_attention_masks)
                    # test_inputs.append(x_test_all)
                    test_inputs.append(x_test_fix_all)
                    test_inputs.append(x_test_all)


                    
                    # val_inputs.append(input_ids)
                    # val_inputs.append(attention_masks)
                    # val_inputs.append(x_test_all)

                    # print("test input ids are ", input_ids.shape)
                    # print("test attention ids are ", attention_masks.shape)
                    # print("test x_test_all ids are ", x_test_all.shape)
                    # print("test val is ", x_val.shape)
                    
                    # if normalize_flag:
                    #     x_test_all = scaler.transform(
                    #         x_test_all.reshape(-1, x_test_all.shape[-1]),
                    #     ).reshape(x_test_all.shape)
                    #     x_test_fix_all = fix_scaler.transform(
                    #         x_test_fix_all.reshape(-1, x_test_fix_all.shape[-1]),
                    #     ).reshape(x_test_fix_all.shape)
                    #     x_test_fix_all = np.where(
                    #         np.isnan(x_test_fix_all), -4, x_test_fix_all,
                    #     )
                    y_test = np.array(y_test_all[:, label_dict[label]], dtype=int)



                    # scale the input
                    # input_scaler = MinMaxScaler()
                    # x_train = input_scaler.fit_transform(x_train)
                    # x_val = input_scaler.transform(x_val)
                    # x_test = input_scaler.transform(x_test)



                    model = get_nn_model(dropout, x_train, input_shape)

                    tf.keras.backend.clear_session()
                    callbacks = [
                        EarlyStopping(
                            monitor='val_loss', patience=patience,
                        ),
                    ]
                    history = model.fit(  # noqa: F841
                        train_inputs, y_train,
                        validation_data=(
                            val_inputs,
                            y_val,
                        ),
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=callbacks,
                        verbose=2,
                    )

                    y_pred = model.predict(
                        test_inputs,
                        batch_size=batch_size,
                    )
                    y_pred = np.array(y_pred).reshape(-1)
                    try:
                        fpr, tpr, _ = metrics.roc_curve(
                            y_test,
                            y_pred,
                            pos_label=1,
                        )
                        auc = metrics.auc(fpr, tpr)
                        print(auc)
                        pd_init[f'fold{fold}_auc'] = auc
                        pd_init[f'fold{fold}_tpr'] = [tpr]
                        pd_init[f'fold{fold}_fpr'] = [fpr]
                        pd_init[f'fold{fold}_y_test'] = [y_test]
                        pd_init[f'fold{fold}_y_pred'] = [y_pred]

                        out_dict[f'fold{fold}_auc'] = auc
                        out_dict[f'fold{fold}_tpr'] = [tpr]
                        out_dict[f'fold{fold}_fpr'] = [fpr]
                        out_dict[f'fold{fold}_y_test'] = [y_test]
                        out_dict[f'fold{fold}_y_pred'] = [y_pred]
                    except KeyError:
                        try:
                            fpr, tpr, _ = metrics.roc_curve(
                                y_test,
                                y_pred,
                                pos_label=1,
                            )
                            auc = metrics.auc(fpr, tpr)
                            print(auc)
                            pd_init[f'fold{fold}_auc'] = auc
                            pd_init[f'fold{fold}_tpr'] = [tpr]
                            pd_init[f'fold{fold}_fpr'] = [fpr]
                            pd_init[f'fold{fold}_y_test'] = y_test
                            pd_init[f'fold{fold}_y_pred'] = y_pred

                            out_dict[f'fold{fold}_auc'] = auc
                            out_dict[f'fold{fold}_tpr'] = [tpr]
                            out_dict[f'fold{fold}_fpr'] = [fpr]
                            out_dict[f'fold{fold}_y_test'] = y_test
                            out_dict[f'fold{fold}_y_pred'] = y_pred
                        except KeyError as e:
                            raise e

                pd_init['avg_auc'] = 0
                out_dict['avg_auc'] = 0
                for i in range(num_folds):
                    pd_init['avg_auc'] += pd_init[f'fold{i}_auc']
                    out_dict['avg_auc'] += out_dict[f'fold{i}_auc']
                pd_init['avg_auc'] /= num_folds
                out_dict['avg_auc'] /= num_folds

                pd_init['std_auc'] = 0
                out_dict['std_auc'] = 0
                for i in range(0, num_folds):
                    pd_init['std_auc'] += (
                        pd_init[f'fold{i}_auc'] -
                        pd_init['avg_auc']
                    )**2
                    out_dict['std_auc'] += (
                        out_dict[f'fold{i}_auc'] - out_dict['avg_auc']
                    )**2
                pd_init['std_auc'] = (pd_init['std_auc'] / num_folds)**(1 / 2)
                out_dict['std_auc'] = (out_dict['std_auc'] / num_folds)**(1 / 2)
                if save_csv:
                    pd_init.to_csv(csv_save_path, index=None)
                if save_joblib:
                    joblib.dump(out_dict, joblib_save_path, compress=3, protocol=2)
                print('mean auc: ' + str(pd_init['avg_auc']))
                
            else:
                hyper = {"seq_len_list": seq_list,
                         "dropout_list": dropout_list}
                
                
                
                for seq in hyper['seq_len_list']:
                    for drop in hyper['dropout_list']:
                        final_df['dropout'].append(drop)
                        final_df['seq_len'].append(seq)
                        final_df['label'].append(label)
                        final_df['split_criterion'].append(split_criterion)

                        for fold in range(num_folds):
                            np.random.seed(fold)
                            random.seed(fold)


                            # collect the inputs for train, validation and test
                            # use only features where flag is True
                            train_inputs = []
                            val_inputs = []
                            test_inputs = []
                            X_train_path = os.path.join(
                                SB_SAT_PATH, f'X_train_{split_criterion}_{fold}.npy',
                            )
                            X_train_fix_path = os.path.join(
                                SB_SAT_PATH, f'X_train_{split_criterion}_{fold}_fix_data.npy',
                            )
                            y_train_path = os.path.join(
                                SB_SAT_PATH, f'y_train_{split_criterion}_{fold}.npy',
                            )
                            x_train_all, y_train_all = np.load(X_train_path).astype(float), np.load(
                                y_train_path, allow_pickle=True,
                            )
                            x_train_fix_all = np.load(X_train_fix_path, allow_pickle= True)
                            x_train_fix_all = x_train_fix_all.astype("float")

                            x_train_fix_all = x_train_fix_all[:, :seq, :]
                            x_train_all = x_train_all[:, :seq, :]
                            # x_train_fix_all = tf.cast(x_train_fix_all, tf.float32)
                            
                            # x_train_fix_all = pd.DataFrame(x_train_fix_all, columns = ['text_list', 'text'])

                            
                            
                            # x_train_fix_postions = x_train_fix_all[:, :, 4]
                            if normalize_flag:
                                scaler = MinMaxScaler()
                                fix_scaler = MinMaxScaler()
                                x_train_all = scaler.fit_transform(
                                    x_train_all.reshape(-1, x_train_all.shape[-1]),
                                ).reshape(x_train_all.shape)
                                x_train_fix_all = fix_scaler.fit_transform(
                                    x_train_fix_all.reshape(-1, x_train_fix_all.shape[-1]),
                                ).reshape(x_train_fix_all.shape)
                                x_train_fix_all = np.where(
                                    np.isnan(x_train_fix_all), -4, x_train_fix_all,
                                )
                            if split_criterion != 'book':
                                outer_cv = KFold(
                                    n_splits=4, shuffle=True,
                                    random_state=fold,
                                )
                            else:
                                outer_cv = KFold(
                                    n_splits=3, shuffle=True,
                                    random_state=fold,
                                )

                            if split_criterion != 'book-page':
                                splitkeys = np.array(
                                    sorted(
                                        list(
                                            set(
                                                y_train_all[
                                                    :,
                                                    split_criterion_dict[split_criterion],
                                                ],
                                            ),
                                        ),
                                    ),
                                )
                            else:
                                splitkeys = y_train_all[:, label_dict[label]]

                            for train_idx, val_idx in outer_cv.split(splitkeys):
                                break

                            if split_criterion != 'book-page':
                                N_train_sub = splitkeys[train_idx]
                                N_test_sub = splitkeys[val_idx]

                                train_idx = np.where(
                                    np.isin(
                                        y_train_all[
                                            :, split_criterion_dict[split_criterion],
                                        ], N_train_sub,
                                    ),
                                )[0]
                                val_idx = np.where(
                                    np.isin(
                                        y_train_all[
                                            :, split_criterion_dict[split_criterion],
                                        ], N_test_sub,
                                    ),
                                )[0]
                            x_train = x_train_all[train_idx]
                            y_train = y_train_all[train_idx]
                            x_val = x_train_all[val_idx]
                            y_val = y_train_all[val_idx]

                            
                            xtr_words = x_train_fix_all[train_idx, :, :]
                            val_words = x_train_fix_all[val_idx, :, :]
                            # y_train_all[val_idx]

                            y_train = np.array(y_train[:, label_dict[label]], dtype=int)
                            y_val = np.array(y_val[:, label_dict[label]], dtype=int)
                            # input_ids, attention_masks = transformer_encode(xtr_words.reset_index(drop = True), input_shape, tokenizer)
                            # val_input_ids, val_attention_masks = transformer_encode(val_words.reset_index(drop = True), input_shape, tokenizer)
                            
                            
                            train_inputs.append(xtr_words)
                            train_inputs.append(x_train)

                            
                            # train_inputs.append(input_ids)
                            # train_inputs.append(attention_masks)
                            # train_inputs.append(x_train)
                            

                            val_inputs.append(val_words)
                            val_inputs.append(x_val)
                            # val_inputs.append(val_input_ids)
                            # val_inputs.append(val_attention_masks)
                            # val_inputs.append(x_val)


                            # print("test val is ", x_val.shape)

                            # Test Data
                            X_test_path = os.path.join(
                                SB_SAT_PATH,
                                f'X_test_{split_criterion}_{fold}.npy',
                            )
                            X_test_fix_path = os.path.join(
                                SB_SAT_PATH,
                                f'X_test_{split_criterion}_{fold}_fix_data.npy',
                            )
                            y_test_path = os.path.join(
                                SB_SAT_PATH,
                                f'y_test_{split_criterion}_{fold}.npy',
                            )
                            x_test_all, y_test_all = np.load(X_test_path).astype(float), np.load(
                                y_test_path, allow_pickle=True,
                            )
                            x_test_fix_all = np.load(X_test_fix_path, allow_pickle=True).astype(float)
                            x_test_fix_all = tf.cast(x_test_fix_all, tf.float32)

                            x_test_fix_all = x_test_fix_all[:, :seq, :]
                            x_test_all = x_test_all[:, :seq, :]
                            # x_test_fix_postions = x_test_fix_all[:, :, 4]
                            # x_test_fix_all = pd.DataFrame(x_test_fix_all, columns = ['text_list', 'text'])
                            # test_input_ids, test_attention_masks = transformer_encode(x_test_fix_all.reset_index(drop = True), input_shape, tokenizer)
                            # test_inputs.append(test_input_ids)
                            # test_inputs.append(test_attention_masks)
                            # test_inputs.append(x_test_all)
                            test_inputs.append(x_test_fix_all)
                            test_inputs.append(x_test_all)


                            
                            # val_inputs.append(input_ids)
                            # val_inputs.append(attention_masks)
                            # val_inputs.append(x_test_all)

                            # print("test input ids are ", input_ids.shape)
                            # print("test attention ids are ", attention_masks.shape)
                            # print("test x_test_all ids are ", x_test_all.shape)
                            # print("test val is ", x_val.shape)
                            
                            # if normalize_flag:
                            #     x_test_all = scaler.transform(
                            #         x_test_all.reshape(-1, x_test_all.shape[-1]),
                            #     ).reshape(x_test_all.shape)
                            #     x_test_fix_all = fix_scaler.transform(
                            #         x_test_fix_all.reshape(-1, x_test_fix_all.shape[-1]),
                            #     ).reshape(x_test_fix_all.shape)
                            #     x_test_fix_all = np.where(
                            #         np.isnan(x_test_fix_all), -4, x_test_fix_all,
                            #     )
                            y_test = np.array(y_test_all[:, label_dict[label]], dtype=int)



                            # scale the input
                            # input_scaler = MinMaxScaler()
                            # x_train = input_scaler.fit_transform(x_train)
                            # x_val = input_scaler.transform(x_val)
                            # x_test = input_scaler.transform(x_test)



                            model = get_nn_model(drop, x_train, seq)

                            tf.keras.backend.clear_session()
                            callbacks = [
                                EarlyStopping(
                                    monitor='val_loss', patience=patience,
                                ),
                            ]
                            history = model.fit(  # noqa: F841
                                train_inputs, y_train,
                                validation_data=(
                                    val_inputs,
                                    y_val,
                                ),
                                batch_size=batch_size,
                                epochs=epochs,
                                callbacks=callbacks,
                                verbose=2,
                            )

                            y_pred = model.predict(
                                test_inputs,
                                batch_size=batch_size,
                            )
                            y_pred = np.array(y_pred).reshape(-1)
                            try:
                                fpr, tpr, _ = metrics.roc_curve(
                                    y_test,
                                    y_pred,
                                    pos_label=1,
                                )
                                auc = metrics.auc(fpr, tpr)
                                print(auc)
                                pd_init[f'fold{fold}_auc'] = auc
                                pd_init[f'fold{fold}_tpr'] = [tpr]
                                pd_init[f'fold{fold}_fpr'] = [fpr]
                                pd_init[f'fold{fold}_y_test'] = [y_test]
                                pd_init[f'fold{fold}_y_pred'] = [y_pred]

                                out_dict[f'fold{fold}_auc'] = auc
                                out_dict[f'fold{fold}_tpr'] = [tpr]
                                out_dict[f'fold{fold}_fpr'] = [fpr]
                                out_dict[f'fold{fold}_y_test'] = [y_test]
                                out_dict[f'fold{fold}_y_pred'] = [y_pred]
                                out_dict[f'fold{fold}_y_pred'] = [y_pred]
                            except KeyError:
                                try:
                                    fpr, tpr, _ = metrics.roc_curve(
                                        y_test,
                                        y_pred,
                                        pos_label=1,
                                    )
                                    auc = metrics.auc(fpr, tpr)
                                    print(auc)
                                    pd_init[f'fold{fold}_auc'] = auc
                                    pd_init[f'fold{fold}_tpr'] = [tpr]
                                    pd_init[f'fold{fold}_fpr'] = [fpr]
                                    pd_init[f'fold{fold}_y_test'] = y_test
                                    pd_init[f'fold{fold}_y_pred'] = y_pred

                                    out_dict[f'fold{fold}_auc'] = auc
                                    out_dict[f'fold{fold}_tpr'] = [tpr]
                                    out_dict[f'fold{fold}_fpr'] = [fpr]
                                    out_dict[f'fold{fold}_y_test'] = y_test
                                    out_dict[f'fold{fold}_y_pred'] = y_pred
                                except KeyError as e:
                                    raise e

                        pd_init['avg_auc'] = 0
                        out_dict['avg_auc'] = 0
                        for i in range(num_folds):
                            pd_init['avg_auc'] += pd_init[f'fold{i}_auc']
                            out_dict['avg_auc'] += out_dict[f'fold{i}_auc']
                        pd_init['avg_auc'] /= num_folds
                        out_dict['avg_auc'] /= num_folds

                        pd_init['std_auc'] = 0
                        out_dict['std_auc'] = 0
                        for i in range(0, num_folds):
                            pd_init['std_auc'] += (
                                pd_init[f'fold{i}_auc'] -
                                pd_init['avg_auc']
                            )**2
                            out_dict['std_auc'] += (
                                out_dict[f'fold{i}_auc'] - out_dict['avg_auc']
                            )**2
                        pd_init['std_auc'] = (pd_init['std_auc'] / num_folds)**(1 / 2)
                        out_dict['std_auc'] = (out_dict['std_auc'] / num_folds)**(1 / 2)
                        
                        final_df['avg_auc'] = pd_init['avg_auc']
                        final_df['std_auc'] = pd_init['std_auc']

                        csv_save_path = f'{save_dir}{model_prefix}_tuning_dropout_{drop}_seqlen_{seq}_{split_criterion}_text_sequence_{label}.csv'  # noqa: E501
                        joblib_save_path = csv_save_path.replace('.csv', '.joblib')
                        if save_csv:
                            pd_init.to_csv(csv_save_path, index=None)
                        if save_joblib:
                            joblib.dump(out_dict, joblib_save_path, compress=3, protocol=2)
                        print('mean auc: ' + str(pd_init['avg_auc']))



    print("Saving the final results ...")
    pd.DataFrame(final_df).to_csv(f"{save_dir}")




            
            
            

def convert_string_to_boolean(input_string):
    if input_string == 'True':
        return True
    return False


def main():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or '2' to filter out warnings as well

    parser = argparse.ArgumentParser()

    # )
    parser.add_argument('-save_dir', '--save_dir', type=str, default='True')
    parser.add_argument('-seq_len', '--seq_len', type=int, default=50)
    parser.add_argument('-dropout', '--dropout', type=float, default=0.3)
    parser.add_argument('-tuning', '--tuning', type=bool, default=False)

    parser.add_argument(
    "-seq_len_list",  # name on the CLI - drop the `--` for positional/required parameters
    nargs="*",  # 0 or more values expected => creates a list
    type=int,
    default=[50, 100],  # default if nothing is provided
    )
    parser.add_argument(
    "-dropout_list",
    nargs="*",
    type=float,  # any type/callable can be used here
    default=[0.1,0.4],
    )

    

    args = parser.parse_args()

    save_dir = args.save_dir
    input_shape = args.seq_len
    dropout = args.dropout
    tuning = args.tuning
    seq_list = args.seq_len_list
    dropout_list = args.dropout_list



    normalize_flag = False
    use_gaze_entropy_features = True

    flag_redo = True
    patience = 7
    batch_size = 256
    epochs = 60

    spit_criterions = ['book-page', 'subj', 'book']
    labels = ['subj_acc_level', 'acc_level', 'native', 'difficulty']
    # labels = ['subj_acc_level', 'acc_level']
    
    model_name = 'nn_Raza'

    feature_names_per_word = [
        'ff',
        'tf',
        'wfc_ff_nomarlized',
        'wfc_tf_normalized',
        'sc_ff_normalized',
        'sc_tf_normalized',
        'ic_ff_normalized',
        'ic_tf_normalized',
        'regression',
        'num_regressions',
        'num_progressions',
        'surprisal',
        'word_len',
        'dependencies_right',
        'dependencies_left',
        'dependency_distance',
    ]



    # train models
    tf.keras.backend.clear_session()
    train_nn(
        spit_criterions=spit_criterions,
        labels=labels,
        feature_names_per_word=feature_names_per_word,
        model_name=model_name,
        input_shape = input_shape,
        dropout = dropout,
        tuning = tuning,
        seq_list = seq_list,
        dropout_list = dropout_list,
        # flag_sequence_bilstm=flag_sequence_bilstm,
        # word_in_fixation_order=word_in_fixation_order,
        # use_reduced_pos_sequence=use_reduced_pos_sequence,
        # use_content_word_sequence=use_content_word_sequence,
        # use_numeric=use_numeric,
        # use_fixation_sequence=use_fixation_sequence,
        flag_redo=flag_redo,
        normalize_flag=normalize_flag,
        use_gaze_entropy_features=use_gaze_entropy_features,
        patience=patience,
        batch_size=batch_size,
        epochs=epochs,
        save_dir=save_dir,
        save_csv=True,
        save_joblib=True,
    )


if __name__ == '__main__':
    raise SystemExit(main())
