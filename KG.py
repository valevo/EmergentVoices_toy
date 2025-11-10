import numpy as np
import pandas as pd


class TFIDF:
    def __init__(self, texts):
        self.N = len(texts)
        self.clean_texts = texts.apply(lambda s: ''.join([i if ord(i) < 128 else '' for i in s]))
        word_lists = self.clean_texts.str.lower().str.split()
        self.tf = Counter(w for ls in word_lists for w in ls)
        doc_f = {}
        for w, c in tqdm(self.tf.most_common()):
            if c > 1:
                doc_f[w] = int(word_lists.apply(lambda ls: w in ls).sum())
            else:
                doc_f[w] = 1
        self.tf = pd.Series(self.tf).sort_index()
        self.doc_f = pd.Series(doc_f).sort_index()
        self.data = pd.concat([self.tf, self.doc_f], axis=1)
        self.data.columns = ["tf", "doc_f"]

    def __call__(self, w):
        try:
            tf = self.data.loc[w].tf
            doc_f = self.data.loc[w].doc_f
            return tf * np.log2(self.N/doc_f)
        except KeyError:
            return 0


class KG:
    def __init__(self, df):
        self.df = df