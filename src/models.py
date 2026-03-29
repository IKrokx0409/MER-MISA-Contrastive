import numpy as np
import random

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig

from utils import to_gpu
from utils import ReverseLayerF


def masked_mean(tensor, mask, dim):
    """Finding the mean along dim"""
    masked = torch.mul(tensor, mask)
    return masked.sum(dim=dim) / mask.sum(dim=dim)

def masked_max(tensor, mask, dim):
    """Finding the max along dim"""
    masked = torch.mul(tensor, mask)
    neg_inf = torch.zeros_like(tensor)
    neg_inf[~mask] = -math.inf
    return (masked + neg_inf).max(dim=dim)



# let's define a simple model that can deal with multimodal variable length sequence
class MISA(nn.Module):
    def __init__(self, config):
        super(MISA, self).__init__()

        self.config = config
        if self.config.data == "mine":
            self.text_size = 768
        else:
            self.text_size = config.embedding_size
            
        self.visual_size = config.visual_size
        self.acoustic_size = config.acoustic_size

        self.image_size = config.image_size # MINE added

        
        # 可学习的 Prompt Pool
        self.prompt_t = nn.Parameter(torch.randn(1, config.hidden_size))
        self.prompt_v = nn.Parameter(torch.randn(1, config.hidden_size))
        self.prompt_a = nn.Parameter(torch.randn(1, config.hidden_size))
        self.prompt_i = nn.Parameter(torch.randn(1, config.hidden_size))


        self.input_sizes = input_sizes = [self.text_size, self.visual_size, self.acoustic_size, self.image_size]
        self.hidden_sizes = hidden_sizes = [int(self.text_size), int(self.visual_size), int(self.acoustic_size), int(self.image_size)]
        self.output_size = output_size = config.num_classes
        self.dropout_rate = dropout_rate = config.dropout
        self.activation = self.config.activation()
        self.tanh = nn.Tanh()
        
        
        rnn = nn.LSTM if self.config.rnncell == "lstm" else nn.GRU
        # defining modules - two layer bidirectional LSTM with layer norm in between

        if self.config.use_bert and self.config.data != "mine":

            # Initializing a BERT bert-base-uncased style configuration
            bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
            self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)
        else:
            if self.config.data != "mine":
                self.embed = nn.Embedding(len(config.word2id), input_sizes[0])
            self.trnn1 = rnn(input_sizes[0], hidden_sizes[0], bidirectional=True)
            self.trnn2 = rnn(2*hidden_sizes[0], hidden_sizes[0], bidirectional=True)
        
        self.vrnn1 = rnn(input_sizes[1], hidden_sizes[1], bidirectional=True)
        self.vrnn2 = rnn(2*hidden_sizes[1], hidden_sizes[1], bidirectional=True)
        
        self.arnn1 = rnn(input_sizes[2], hidden_sizes[2], bidirectional=True)
        self.arnn2 = rnn(2*hidden_sizes[2], hidden_sizes[2], bidirectional=True)

        self.irnn1 = rnn(768, config.hidden_size, bidirectional=True)
        self.irnn2 = rnn(2*config.hidden_size, config.hidden_size, bidirectional=True)
        self.ilayer_norm = nn.LayerNorm((config.hidden_size*2,))


        ##########################################
        # mapping modalities to same sized space
        ##########################################
        if self.config.use_bert and self.config.data != "mine":
            self.project_t = nn.Sequential()
            self.project_t.add_module('project_t', nn.Linear(in_features=768, out_features=config.hidden_size))
            self.project_t.add_module('project_t_activation', self.activation)
            self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))
        else:
            self.project_t = nn.Sequential()
            self.project_t.add_module('project_t', nn.Linear(in_features=hidden_sizes[0]*4, out_features=config.hidden_size))
            self.project_t.add_module('project_t_activation', self.activation)
            self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_v = nn.Sequential()
        self.project_v.add_module('project_v', nn.Linear(in_features=hidden_sizes[1]*4, out_features=config.hidden_size))
        self.project_v.add_module('project_v_activation', self.activation)
        self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a', nn.Linear(in_features=hidden_sizes[2]*4, out_features=config.hidden_size))
        self.project_a.add_module('project_a_activation', self.activation)
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_i = nn.Sequential()
        self.project_i.add_module('project_i', nn.Linear(in_features=config.hidden_size*4, out_features=config.hidden_size)) # MINE 特征是768
        self.project_i.add_module('project_i_activation', self.activation)
        self.project_i.add_module('project_i_layer_norm', nn.LayerNorm(config.hidden_size))

        ##########################################
        # private encoders
        ##########################################
        self.private_t = nn.Sequential()
        self.private_t.add_module('private_t_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_t.add_module('private_t_activation_1', nn.Sigmoid())
        
        self.private_v = nn.Sequential()
        self.private_v.add_module('private_v_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_v.add_module('private_v_activation_1', nn.Sigmoid())
        
        self.private_a = nn.Sequential()
        self.private_a.add_module('private_a_3', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_a.add_module('private_a_activation_3', nn.Sigmoid())

        self.private_i = nn.Sequential()
        self.private_i.add_module('private_i_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_i.add_module('private_i_activation_1', nn.Sigmoid())
        

        ##########################################
        # shared encoder
        ##########################################
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())


        ##########################################
        # reconstruct
        ##########################################
        self.recon_t = nn.Sequential()
        self.recon_t.add_module('recon_t_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.recon_v = nn.Sequential()
        self.recon_v.add_module('recon_v_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.recon_a = nn.Sequential()
        self.recon_a.add_module('recon_a_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.recon_i = nn.Sequential()
        self.recon_i.add_module('recon_i_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))


        ##########################################
        # shared space adversarial discriminator
        ##########################################
        if not self.config.use_cmd_sim:
            self.discriminator = nn.Sequential()
            self.discriminator.add_module('discriminator_layer_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
            self.discriminator.add_module('discriminator_layer_1_activation', self.activation)
            self.discriminator.add_module('discriminator_layer_1_dropout', nn.Dropout(dropout_rate))
            self.discriminator.add_module('discriminator_layer_2', nn.Linear(in_features=config.hidden_size, out_features=len(hidden_sizes)))

        ##########################################
        # shared-private collaborative discriminator
        ##########################################

        self.sp_discriminator = nn.Sequential()
        self.sp_discriminator.add_module('sp_discriminator_layer_1', nn.Linear(in_features=config.hidden_size, out_features=4))



        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=self.config.hidden_size*8, out_features=self.config.hidden_size*3))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        self.fusion.add_module('fusion_layer_1_activation', self.activation)
        # self.fusion.add_module('fusion_layer_3', nn.Linear(in_features=self.config.hidden_size*3, out_features= output_size))
        self.emotion_head = nn.Linear(self.config.hidden_size*3, config.num_classes_emotion) # 11类
        self.intent_head = nn.Linear(self.config.hidden_size*3, config.num_classes_intent)   # 21类

        
        self.tlayer_norm = nn.LayerNorm((hidden_sizes[0]*2,))
        self.vlayer_norm = nn.LayerNorm((hidden_sizes[1]*2,))
        self.alayer_norm = nn.LayerNorm((hidden_sizes[2]*2,))


        encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.hidden_size, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        

        
    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        packed_sequence = pack_padded_sequence(sequence, lengths.cpu())

        if self.config.rnncell == "lstm":
            packed_h1, (final_h1, _) = rnn1(packed_sequence)
        else:
            packed_h1, final_h1 = rnn1(packed_sequence)

        padded_h1, _ = pad_packed_sequence(packed_h1)
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths.cpu())

        if self.config.rnncell == "lstm":
            _, (final_h2, _) = rnn2(packed_normed_h1)
        else:
            _, final_h2 = rnn2(packed_normed_h1)

        return final_h1, final_h2

    
    # def alignment(self, sentences, visual, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask):
        
    #     batch_size = lengths.size(0)
        
    #     if self.config.use_bert:
    #         bert_output = self.bertmodel(input_ids=bert_sent, 
    #                                      attention_mask=bert_sent_mask, 
    #                                      token_type_ids=bert_sent_type)      

    #         bert_output = bert_output[0]

    #         # masked mean
    #         masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)
    #         mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)  
    #         bert_output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len

    #         utterance_text = bert_output
    #     else:
    #         # extract features from text modality
    #         sentences = self.embed(sentences)
    #         final_h1t, final_h2t = self.extract_features(sentences, lengths, self.trnn1, self.trnn2, self.tlayer_norm)
    #         utterance_text = torch.cat((final_h1t, final_h2t), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)



    #     # extract features from visual modality
    #     final_h1v, final_h2v = self.extract_features(visual, lengths, self.vrnn1, self.vrnn2, self.vlayer_norm)
    #     utterance_video = torch.cat((final_h1v, final_h2v), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

    #     # extract features from acoustic modality
    #     final_h1a, final_h2a = self.extract_features(acoustic, lengths, self.arnn1, self.arnn2, self.alayer_norm)
    #     utterance_audio = torch.cat((final_h1a, final_h2a), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

    #     # Shared-private encoders
    #     self.shared_private(utterance_text, utterance_video, utterance_audio)


    #     if not self.config.use_cmd_sim:
    #         # discriminator
    #         reversed_shared_code_t = ReverseLayerF.apply(self.utt_shared_t, self.config.reverse_grad_weight)
    #         reversed_shared_code_v = ReverseLayerF.apply(self.utt_shared_v, self.config.reverse_grad_weight)
    #         reversed_shared_code_a = ReverseLayerF.apply(self.utt_shared_a, self.config.reverse_grad_weight)

    #         self.domain_label_t = self.discriminator(reversed_shared_code_t)
    #         self.domain_label_v = self.discriminator(reversed_shared_code_v)
    #         self.domain_label_a = self.discriminator(reversed_shared_code_a)
    #     else:
    #         self.domain_label_t = None
    #         self.domain_label_v = None
    #         self.domain_label_a = None


    #     self.shared_or_private_p_t = self.sp_discriminator(self.utt_private_t)
    #     self.shared_or_private_p_v = self.sp_discriminator(self.utt_private_v)
    #     self.shared_or_private_p_a = self.sp_discriminator(self.utt_private_a)
    #     self.shared_or_private_s = self.sp_discriminator( (self.utt_shared_t + self.utt_shared_v + self.utt_shared_a)/3.0 )
        
    #     # For reconstruction
    #     self.reconstruct()
        
    #     # 1-LAYER TRANSFORMER FUSION
    #     h = torch.stack((self.utt_private_t, self.utt_private_v, self.utt_private_a, self.utt_shared_t, self.utt_shared_v,  self.utt_shared_a), dim=0)
    #     h = self.transformer_encoder(h)
    #     h = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5]), dim=1)
    #     o = self.fusion(h)
    #     return o
    
    def alignment(self, text, video, audio, image, mask):
        batch_size = text.size(1)
        
        # 为了使用 extract_features，我们需要为长度为 1 的模态创建全 1 的长度向量
        # 也可以直接复用传入的 lengths (如果所有模态在 data_loader 里都对齐了)
        ones_lengths = torch.ones(batch_size, dtype=torch.long).to(text.device)
        audio_lengths = torch.full((batch_size,), 60, dtype=torch.long).to(text.device)

        # 🌟 第一阶段：原样调用 extract_features (RNN 建模)
        # Text (Seq=1)
        f1t, f2t = self.extract_features(text, ones_lengths, self.trnn1, self.trnn2, self.tlayer_norm)
        utt_t = torch.cat((f1t, f2t), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        # Video (Seq=1)
        f1v, f2v = self.extract_features(video, ones_lengths, self.vrnn1, self.vrnn2, self.vlayer_norm)
        utt_v = torch.cat((f1v, f2v), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        # Audio (Seq=60)
        f1a, f2a = self.extract_features(audio, audio_lengths, self.arnn1, self.arnn2, self.alayer_norm)
        utt_a = torch.cat((f1a, f2a), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        # Image (Seq=1)
        f1i, f2i = self.extract_features(image, ones_lengths, self.irnn1, self.irnn2, self.ilayer_norm)
        utt_i = torch.cat((f1i, f2i), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        # 🌟 第二阶段：Shared-Private 投影
        # 这里的 project 函数会将 RNN 出来的维度压缩到 config.hidden_size
        self.utt_t_orig = utt_t = self.project_t(utt_t)
        self.utt_v_orig = utt_v = self.project_v(utt_v)
        self.utt_a_orig = utt_a = self.project_a(utt_a)
        self.utt_i_orig = utt_i = self.project_i(utt_i)

        # 🌟 第三阶段：缺失模态补全 (关键插入点)
        # 在投影之后、分解之前，用 Prompt 替换缺失部分
        for b in range(batch_size):
            if mask[b, 0] == 0: utt_t[b] = self.prompt_t
            if mask[b, 1] == 0: utt_a[b] = self.prompt_a
            if mask[b, 2] == 0: utt_v[b] = self.prompt_v
            if mask[b, 3] == 0: utt_i[b] = self.prompt_i

        # 🌟 第四阶段：Private-Shared 分解
        self.utt_private_t = self.private_t(utt_t)
        self.utt_private_v = self.private_v(utt_v)
        self.utt_private_a = self.private_a(utt_a)
        self.utt_private_i = self.private_i(utt_i)

        self.utt_shared_t = self.shared(utt_t)
        self.utt_shared_v = self.shared(utt_v)
        self.utt_shared_a = self.shared(utt_a)
        self.utt_shared_i = self.shared(utt_i)

        if not self.config.use_cmd_sim:
            reversed_shared_code_t = ReverseLayerF.apply(self.utt_shared_t, self.config.reverse_grad_weight)
            reversed_shared_code_v = ReverseLayerF.apply(self.utt_shared_v, self.config.reverse_grad_weight)
            reversed_shared_code_a = ReverseLayerF.apply(self.utt_shared_a, self.config.reverse_grad_weight)
            reversed_shared_code_i = ReverseLayerF.apply(self.utt_shared_i, self.config.reverse_grad_weight) # 新增

            self.domain_label_t = self.discriminator(reversed_shared_code_t)
            self.domain_label_v = self.discriminator(reversed_shared_code_v)
            self.domain_label_a = self.discriminator(reversed_shared_code_a)
            self.domain_label_i = self.discriminator(reversed_shared_code_i) # 新增
        else:
            self.domain_label_t = self.domain_label_v = self.domain_label_a = self.domain_label_i = None

        self.shared_or_private_p_t = self.sp_discriminator(self.utt_private_t)
        self.shared_or_private_p_v = self.sp_discriminator(self.utt_private_v)
        self.shared_or_private_p_a = self.sp_discriminator(self.utt_private_a)
        self.shared_or_private_p_i = self.sp_discriminator(self.utt_private_i) # 新增
        self.shared_or_private_s = self.sp_discriminator( (self.utt_shared_t + self.utt_shared_v + self.utt_shared_a + self.utt_shared_i)/4.0 ) # 改为除以 4.0
        
        self.reconstruct()

        # 🌟 第五阶段：Transformer 融合与双头输出
        h = torch.stack((self.utt_private_t, self.utt_private_v, self.utt_private_a, self.utt_private_i,
                         self.utt_shared_t, self.utt_shared_v, self.utt_shared_a, self.utt_shared_i), dim=0)
        h = self.transformer_encoder(h)
        # 拼接 8 个分量的特征
        h_flat = torch.cat([h[j] for j in range(8)], dim=1)
        
        feat_fused = self.fusion(h_flat)
        emo_out = self.emotion_head(feat_fused)
        intent_out = self.intent_head(feat_fused)

        return emo_out, intent_out


    def reconstruct(self,):

        self.utt_t = (self.utt_private_t + self.utt_shared_t)
        self.utt_v = (self.utt_private_v + self.utt_shared_v)
        self.utt_a = (self.utt_private_a + self.utt_shared_a)
        self.utt_i = (self.utt_private_i + self.utt_shared_i)

        self.utt_t_recon = self.recon_t(self.utt_t)
        self.utt_v_recon = self.recon_v(self.utt_v)
        self.utt_a_recon = self.recon_a(self.utt_a)
        self.utt_i_recon = self.recon_i(self.utt_i)


    def shared_private(self, utterance_t, utterance_v, utterance_a):
        
        # Projecting to same sized space
        self.utt_t_orig = utterance_t = self.project_t(utterance_t)
        self.utt_v_orig = utterance_v = self.project_v(utterance_v)
        self.utt_a_orig = utterance_a = self.project_a(utterance_a)


        # Private-shared components
        self.utt_private_t = self.private_t(utterance_t)
        self.utt_private_v = self.private_v(utterance_v)
        self.utt_private_a = self.private_a(utterance_a)

        self.utt_shared_t = self.shared(utterance_t)
        self.utt_shared_v = self.shared(utterance_v)
        self.utt_shared_a = self.shared(utterance_a)


    # def forward(self, sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask):
    #     batch_size = lengths.size(0)
    #     o = self.alignment(sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask)
    #     return o

    def forward(self, text, video, audio, image, mask):
        emo_out, intent_out = self.alignment(text, video, audio, image, mask)
        return emo_out, intent_out