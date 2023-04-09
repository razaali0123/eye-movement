import numpy as np

def craft_features(fixation_list, word_list):
    word_len = [len(i) for i in word_list]
    word_occ = [len(i) for i in fixation_list]
    return np.array([word_len, word_occ])
    