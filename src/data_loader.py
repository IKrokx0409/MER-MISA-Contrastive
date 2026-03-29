import random
import numpy as np
from tqdm import tqdm_notebook
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import *

# add mine
from create_dataset import MOSI, MOSEI, UR_FUNNY, MINE, PAD, UNK


bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class MSADataset(Dataset):
    def __init__(self, config):

        ## Fetch dataset
        if "mosi" in str(config.data_dir).lower():
            dataset = MOSI(config)
        elif "mosei" in str(config.data_dir).lower():
            dataset = MOSEI(config)
        elif "ur_funny" in str(config.data_dir).lower():
            dataset = UR_FUNNY(config)
        elif "mine" in str(config.data_dir).lower():
            dataset = MINE(config)
        else:
            print("Dataset not defined correctly")
            exit()
        
        self.data, self.word2id, self.pretrained_emb = dataset.get_data(config.mode)
        self.len = len(self.data)

        if "mine" in str(config.data_dir).lower():
            # MINE: data[0][0] 是 (t, v, a, i, mask)
            config.text_size = self.data[0][0][0].shape[-1]     # 768
            config.visual_size = self.data[0][0][1].shape[-1]   # 512
            config.acoustic_size = self.data[0][0][2].shape[-1] # 768
            config.image_size = self.data[0][0][3].shape[-1]    # 768
            config.word2id = None
            config.pretrained_emb = None

        else:
            config.visual_size = self.data[0][0][1].shape[1]
            config.acoustic_size = self.data[0][0][2].shape[1]

            config.word2id = self.word2id
            config.pretrained_emb = self.pretrained_emb


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len



def get_loader(config, shuffle=True):
    """Load DataLoader of given DialogDataset"""

    dataset = MSADataset(config)
    
    print(config.mode)
    config.data_len = len(dataset)


    def collate_fn(batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''
        # for later use we sort the batch in descending order of length
        batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)
        
        # get the data out of the batch - use pad sequence util functions from PyTorch to pad things


        labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)
        sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=PAD)
        visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch])
        acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch])

        ## BERT-based features input prep

        SENT_LEN = sentences.size(0)
        # Create bert indices using tokenizer

        bert_details = []
        for sample in batch:
            text = " ".join(sample[0][3])
            encoded_bert_sent = bert_tokenizer(
                text, max_length=SENT_LEN+2, add_special_tokens=True, padding='max_length', truncation=True)
            bert_details.append(encoded_bert_sent)
            # delete encode_plus


        # Bert things are batch_first
        bert_sentences = torch.LongTensor([sample["input_ids"] for sample in bert_details])
        bert_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in bert_details])
        bert_sentence_att_mask = torch.LongTensor([sample["attention_mask"] for sample in bert_details])


        # lengths are useful later in using RNNs
        lengths = torch.LongTensor([sample[0][0].shape[0] for sample in batch])

        return sentences, visual, acoustic, labels, lengths, bert_sentences, bert_sentence_types, bert_sentence_att_mask

    # MINE dataset collate function
    def collate_fn_mine(batch):
        t_batch, v_batch, a_batch, i_batch, mask_batch = [], [], [], [], []
        emo_labels = []
        # 创建全 0 意图矩阵做 Multi-hot
        intent_labels = torch.zeros(len(batch), config.num_classes_intent)
        ids = []

        def clean_feat(feat):
            feat = np.nan_to_num(feat) # 把潜在的 nan 变成 0
            # 压制方差，防止 LSTM 梯度爆炸
            feat = (feat - np.mean(feat, axis=0, keepdims=True)) / (np.std(feat, axis=0, keepdims=True) + 1e-6)
            return feat

        for b_idx, sample in enumerate(batch):
            feats, emo, intent, item_id = sample
            t, v, a, img, mask = feats
            t = clean_feat(t)
            v = clean_feat(v)
            a = clean_feat(a)
            img = clean_feat(img)
            
            # Text, Video, Image 升维变序列
            t_batch.append(torch.FloatTensor(t.copy()).unsqueeze(0)) 
            v_batch.append(torch.FloatTensor(v.copy()).unsqueeze(0))
            i_batch.append(torch.FloatTensor(img.copy()).unsqueeze(0))
            # Audio 直接转 Tensor
            a_batch.append(torch.FloatTensor(a.copy()))
            mask_batch.append(torch.FloatTensor(mask.copy()))
            
            emo_labels.append(emo)
            if len(intent) > 0:
                intent_labels[b_idx, intent] = 1.0 # Multi-hot 激活
            ids.append(item_id)
            
        t_tensor = pad_sequence(t_batch)
        v_tensor = pad_sequence(v_batch)
        i_tensor = pad_sequence(i_batch)
        a_tensor = pad_sequence(a_batch)
        
        emo_tensor = torch.LongTensor(emo_labels)
        mask_tensor = torch.stack(mask_batch)

        return t_tensor, v_tensor, a_tensor, i_tensor, emo_tensor, intent_labels, mask_tensor, ids
        

    is_mine = "mine" in str(config.data_dir).lower()
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn_mine if is_mine else collate_fn)

    return data_loader
