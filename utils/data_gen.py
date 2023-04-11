import numpy as np
import pandas as pd
from tqdm import tqdm


def adding_word_length(x):
    if pd.isna(x['CURRENT_FIX_INTEREST_AREA_LABEL']):
        return 0
    else:
        return len(x.CURRENT_FIX_INTEREST_AREA_LABEL)
    

def data_prep(df, sc, seq_len = 200):
    subj = df.subj.unique()
    page_book = df.page_name.unique()
    a = len(page_book)
    z = len(subj) * a
    cols_to_drop = ['pnr',  'book', 'language', 'acc_level', 'subj_acc_level','sac_angle', 'sac_amplitude', 'sac_velcity',
    'sac_blink', 'native', 'difficulty'] 
    df = df.drop(cols_to_drop, axis = 1)
    num_features = df.shape[1]-6 # will remove [acc, subj_acc, subj, page_book, 'word'] + book_name later
    words = []
    matrix = np.zeros(shape = (z, seq_len, num_features))

    target_acc = []
    target_subj_acc = []
    mask = []
    label_arr = np.empty((0, sc.shape[1]))
    


    for cnt, pb in enumerate(tqdm(page_book)):
        for cnt_person, s in enumerate(subj):
            temp = df[(df.subj == s) & (df.page_name == pb)]
            tmp_label = sc.loc[
                sc.subj ==
                s
            ].loc[sc.book == pb.split('-')[1]]
            label_arr = np.vstack([label_arr, tmp_label])
            if temp.shape[0] == 0:
                mask.append((None, None, None))
                target_acc.append(None)
                target_subj_acc.append(None)
                words.append(None)
                continue
            temp = temp.sort_values("word_loc")
            mask.append((s,temp['book_name'].iloc[0],pb))
            temp = temp.drop("book_name", axis = 1)
            
            target_acc.append(temp['acc'].iloc[0])
            target_subj_acc.append(temp['subj_acc'].iloc[0])
            words.append(temp['word'].tolist())
            temp = temp.drop(['acc', 'subj_acc', 'page_name', 'subj', 'word'], axis = 1)
            if temp.shape[0] >= seq_len:
                temp = (temp.iloc[:seq_len, :]).to_numpy()
                words[-1] = words[-1][:seq_len]
            else:
                diff = seq_len - temp.shape[0]

                words[-1] = words[-1] + [' ']*diff

                to_add = np.zeros(shape= (diff, temp.shape[1]))

                temp = temp.to_numpy()

                temp = np.append(temp, to_add, axis=0)


            matrix[cnt*a + cnt_person, :, :] = temp
    return matrix, target_acc, target_subj_acc,  mask, words, label_arr

def data_gen(seq_length):
    df = pd.read_csv("SB-SAT/fixation/18sat_fixfinal.csv")
    label = pd.read_csv("SB-SAT/fixation/18sat_labels.csv")
    label_dict = {label: idx for idx, label in enumerate(label.columns.tolist())}
    
    ## old label_arr
    
    labelcols = [
        'subj', 'book', 'acc_level', 'subj_acc_level', 'confidence', 'difficulty', 'familiarity',
        'interest', 'pressured', 'sleepiness', 'sleephours', 'sex', 'native',
    ]
    sc = label.copy()
    sc['sex'] = sc['sex'].replace(['F', 'M'], [1, 0])
    binarycols = ('recognition', 'sex', 'native')
    subsetcols = [c for c in labelcols if c not in binarycols]
    sc[subsetcols] = sc[subsetcols].replace([0, 1, 2, 3], [0, 0, 1, 1])
    
    
    
    
    
    ###
    df['word_length'] = df.apply(adding_word_length, axis = 1)
    
    df.rename(columns= {"RECORDING_SESSION_LABEL": "pnr", "CURRENT_FIX_X": "xx", "CURRENT_FIX_Y": "yy", "CURRENT_FIX_PUPIL": "pupil", \
                    "CURRENT_FIX_DURATION":"duration", "CURRENT_FIX_INTEREST_AREA_ID":"word_loc", "CURRENT_FIX_INTEREST_AREA_LABEL": "word"}, inplace = True)

    word_occ = df[['pnr','page_name' ,'word_loc','word', 'word_length']].groupby(['pnr','page_name' ,'word_loc','word'], as_index= False).count()
    word_occ.rename(columns = {"word_length":"word_occurance_count"}, inplace=True)



    df = df.merge(word_occ, on= ['pnr','page_name' ,'word_loc','word'])
    
    df.rename(columns= {"CURRENT_FIX_INTEREST_AREA_PIXEL_AREA": "pixel_area", "CURRENT_FIX_INTEREST_AREA_DWELL_TIME": "dwell_time", "PREVIOUS_SAC_DIRECTION": "sac_direction", "PREVIOUS_SAC_ANGLE": "sac_angle", \
                    "PREVIOUS_SAC_AMPLITUDE":"sac_amplitude", "PREVIOUS_SAC_AVG_VELOCITY":"sac_velcity", "PREVIOUS_SAC_CONTAINS_BLINK": "sac_blink"}, inplace = True)
    
    selected_cols = ['pnr', 'xx', 'yy', 'pupil', 'duration', 'RT',  'word_length', 'word_occurance_count', 'book_name', 'page_name', 'word', 'word_loc', \
                 'pixel_area', 'dwell_time','sac_direction', 'sac_angle', 'sac_amplitude', 'sac_velcity','sac_blink']

    df_cognitive = df[selected_cols] \
    .merge(label, left_on=['pnr', 'book_name'], right_on=['subj', 'book'], how = 'inner' )
    
    df_cognitive['sex'] = df_cognitive['sex'].map({'M':1, 'F':0})
    data = df_cognitive.copy()
    data.to_csv("/home/raza/repo/etra-reading-comprehension/SB-SAT/fixation/df_cognitive.csv",index= False)
    
    dummies_sac_direction = pd.get_dummies(data.sac_direction)
    data = pd.concat([data, dummies_sac_direction], axis = 1).drop("sac_direction", axis = 1)
    
    # seq_length = 50
    mat, target , target_subj, mask, words, label_arr = data_prep(data, seq_len = seq_length, sc = sc)
    word_df = pd.DataFrame({"text_lst": words})
    word_df['text'] = word_df.text_lst.apply(lambda x: ' '.join(x) if x is not None else x)
    mask = np.array(mask)
    target = np.array(target)
    target_subj = np.array(target_subj)
    ii = np.invert(pd.isnull(target))
    new_target = target[ii]
    new_target_subj = target_subj[ii]
    new_mat = mat[ii, :, :]
    new_mask = mask[ii, :]
    new_word_df = word_df.iloc[ii, :]
    label_arr = label_arr[ii, :]
    return new_mat, new_target , new_target_subj, new_mask, new_word_df, label_dict, label_arr
    



        