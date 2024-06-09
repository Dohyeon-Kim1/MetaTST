import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class TextStyleDataset(Dataset):
    def __init__(self, args):
        self.mode = args.mode
        en2kr = {"formal": "문어체", "informal": "구어체", "android": "안드로이드", "azae": "아저씨", "chat": "채팅", "choding": "초등학생",
                 "emoticon": "이모티콘", "enfp": "enfp", "gentle": "신사", "halbae": "할아버지", "halmae": "할머니", "joongding": "중학생",
                 "king": "왕", "naruto": "나루토", "seonbi": "선비", "sosim": "소심한", "translator": "번역기"}
        df = (pd.read_csv("smilestyle_dataset.tsv", delimiter="\t")).dropna()
        df.columns = [en2kr[col] for col in df.columns]
        self.columns = df.columns[:-3] if self.mode == "train" else df.columns[-3:]
        self.data = df[self.columns]

        if self.mode != "train":
            self.origin = df[["문어체"]]

    def __len__(self):
        return len(self.data)*len(self.columns)
    
    def __getitem__(self, idx):
        if self.mode == "train":
            row = idx % len(self.data)
            col = idx // len(self.data)
            t_col = random.randint(0,len(self.columns)-1)
            return self.data.iloc[row, col], self.data.iloc[row, t_col], self.columns[t_col]
        
        else:
            row = idx % len(self.data)
            col = idx // len(self.data)
            return self.origin.iloc[row,0], self.data.iloc[row,col], self.columns[col]


class MetaTextStyleDataset(Dataset):
    def __init__(self, args):
        self.mode = args.mode
        en2kr = {"formal": "문어체", "informal": "구어체", "android": "안드로이드", "azae": "아저씨", "chat": "채팅", "choding": "초등학생",
                 "emoticon": "이모티콘", "enfp": "enfp", "gentle": "신사", "halbae": "할아버지", "halmae": "할머니", "joongding": "중학생",
                 "king": "왕", "naruto": "나루토", "seonbi": "선비", "sosim": "소심한", "translator": "번역기"}
        self.data = (pd.read_csv("smilestyle_dataset.tsv", delimiter="\t")).dropna()
        self.data.columns = [en2kr[col] for col in self.data.columns]
        self.columns = self.data.columns[:-3] if args.mode == "train" else self.data.columns[-3:]
        self.idxs = np.array(range(len(self.data)))
        
        self.n_shot = args.n_shot
        self.n_qry = args.n_qry

        if self.mode != "train":
            self.origin = self.data["문어체"]

    def __len__(self):
        return len(self.data)*len(self.columns)

    def __getitem__(self, idx):
        if self.mode == "train":
            columns = np.random.choice(self.columns, 2)
            
            np.random.shuffle(self.idxs)
            x_spt1 = self.data[columns[0]].iloc[self.idxs[:self.n_shot]].values.tolist()
            x_qry1 = self.data[columns[0]].iloc[self.idxs[self.n_shot:self.n_shot+self.n_qry]].values.tolist()
            x_spt2 = self.data[columns[1]].iloc[self.idxs[:self.n_shot]].values.tolist()
            x_qry2 = self.data[columns[1]].iloc[self.idxs[self.n_shot:self.n_shot+self.n_qry]].values.tolist()

            return (x_spt1, x_spt2), (x_qry1, x_qry2), columns[1]

        else:
            column = self.columns[idx//len(self.data)]

            np.random.shuffle(self.idxs)
            x_spt1 = self.origin.iloc[self.idxs[:self.n_shot]].values.tolist()
            x_qry1 = self.origin.iloc[[idx%len(self.data)]].values.tolist()
            x_spt2 = self.data[column].iloc[self.idxs[:self.n_shot]].values.tolist()
            x_qry2 = self.data[column].iloc[[idx%len(self.data)]].values.tolist()

            return (x_spt1, x_spt2), (x_qry1, x_qry2), column