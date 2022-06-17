import csv
import pandas as pd 
import numpy as np
def Proba(Unicode, Vocabrary, Word):
    ProbaUnicode = []
    df = pd.read_csv('./kana_prob/'+ Word+'.csv', index_col=5)
    for i in range(48):
        A = df[df["char2"] == Vocabrary[i]]
        C = Unicode[i] * (10+A.index[0])
        ProbaUnicode.append(C)
    return np.array(ProbaUnicode)