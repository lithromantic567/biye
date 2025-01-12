import torch
from torch import nn
from util import *
from Param import *
import numpy as np
import os
from RoutePlan import *
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from torch.distributions import Categorical
from arg_parser import parse_arguments
args = parse_arguments()

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_length,dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        positions = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
        pos_enc = torch.zeros((max_length, hidden_dim))

        pos_enc[:, 0::2] = torch.sin(positions * div_term)
        pos_enc[:, 1::2] = torch.cos(positions * div_term)

        self.register_buffer('pe', pos_enc.unsqueeze(0))

    def forward(self, x, i=None, method=None):
        # 输入形状: (batch_size, seq_len, hidden_dim)
        if method=="decoder":
            x = x + self.pe[:, i]
        else:
            x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
class ELG_A(nn.Module):
    def __init__(self):
        super(ELG_A, self).__init__()
        self.voc_embedding = nn.Embedding(Param.voc_size, Param.room_emb_size)  # start token, and end token
        self.positional_encoding = PositionalEncoding(Param.room_emb_size, Param.max_sent_len)
        self.transformer = TransformerDecoder(
            TransformerDecoderLayer(Param.room_emb_size, Param.nhead,dim_feedforward=32),
            num_layers=3
        )
        #self.positional_encoding=PositionalEncoding(Param.room_emb_size,Param.max_sent_len)
        self.emb2idx = nn.Sequential(
            nn.Linear(Param.room_emb_size, Param.voc_size),
            nn.Softmax(dim=1)
        )
        
    def forward(self, cur_room_emb, max_length, choose_token_method="sample"):
        spoken_token_prob = []
        spoken_token = []
        next_token_prob = []
        #patch_emb=cur_room_emb.reshape(Param.batch_size,Param.nhead,Param.patch_per_size)
        #cls_tokens=self.cls_token.repeat(Param.batch_size,1,1)
        #patch_emb=torch.cat([cls_tokens,patch_emb],dim=1)
        #patch_emb+=self.positions
        #output=self.transformer(patch_emb)
        
        assert cur_room_emb.shape[1] ==Param.room_emb_size
        hx= cur_room_emb.unsqueeze(0)
        #(1,50,50)
        #hx = self.transformer(cur_room_emb.unsqueeze(0)).squeeze(0)
        device = self.voc_embedding.weight.device
        token_before = self.voc_embedding(torch.LongTensor([[Param.sos_idx for _ in range(cur_room_emb.shape[0])]]).to(device))
        #(1,50,50)    
        for i in range(max_length):
            output = self.transformer(token_before, hx)         
            next_token_pred = self.emb2idx(output[-1,:,:].squeeze(0))
            if choose_token_method == "greedy":
                token_idx = torch.argmax(next_token_pred, dim=1) 
                # 计算贪婪选择的 token 的 log 概率
                log_prob = torch.log_softmax(next_token_pred, dim=1)  # 将概率转化为 log 概率
                chosen_log_prob = log_prob[torch.arange(log_prob.size(0)), token_idx]  # 提取选择的 token 的 log 概率
                spoken_token_prob.append(chosen_log_prob)  # 记录 log 概率
            elif choose_token_method == "sample":
                next_token_prob.append(next_token_pred)
                token_sampler = torch.distributions.Categorical(next_token_pred)
                token_idx = token_sampler.sample()
                # record actions
                spoken_token_prob.append(token_sampler.log_prob(token_idx))
            spoken_token.append(token_idx)
            spoken_token_emb = self.voc_embedding(token_idx).unsqueeze(1)
            spoken_token_pos = self.positional_encoding(spoken_token_emb,i,method="decoder")
            spoken_token_pos = torch.transpose(spoken_token_pos,0,1)
            token_before = torch.cat((token_before,spoken_token_pos),dim=0)
            #token_before = self.voc_embedding(token_idx).unsqueeze(0)
        spoken_token = torch.reshape(torch.cat(spoken_token, axis=0), (Param.max_sent_len, cur_room_emb.shape[0]))
        spoken_token = torch.transpose(spoken_token, 0, 1)
        assert spoken_token.shape[0] == cur_room_emb.shape[0]
        return spoken_token, spoken_token_prob 
    
    def cal_loss(self, spoken_token_prob, reward):
        spoken_token_prob = torch.stack(spoken_token_prob, dim=0).t()
        loss = -torch.sum(spoken_token_prob * reward.unsqueeze(1),dim=0)
        return loss.mean()

class ELU_B(nn.Module):
    def __init__(self):
        super(ELU_B, self).__init__()
        self.voc_embedding = nn.Embedding(Param.voc_size + 2, Param.room_emb_size)
        self.sent_encoder = TransformerEncoder(
            TransformerEncoderLayer(Param.room_emb_size, Param.nhead,dim_feedforward=32),
            num_layers=3
        )
        self.emb2idx = nn.Sequential(
            nn.Linear(Param.room_emb_size, Param.room_emb_size)
        )
        #self.positions=nn.Parameter(torch.randn(Param.nhead,Param.voc_emb_size))
        self.softmax = nn.Softmax(dim=1)
        self.positional_encoding=PositionalEncoding(Param.room_emb_size,Param.max_sent_len)
        self.fc_type=nn.Sequential(
            nn.Linear(Param.room_emb_size,3)
            )
        self.fc_color=nn.Sequential(
            nn.Linear(Param.room_emb_size,6)
            )
        

    def _encode_message(self, message):
        # message -> (batch, message len)
        msg_shape = message.shape
        #reshaped_msg = message.reshape((msg_shape[0] * msg_shape[1], msg_shape[2])).long()
        # msg_embs -> (batch, message len, voc_emb_size)
        msg_embs = self.voc_embedding(message)
        msg_embs = self.positional_encoding(msg_embs)
        msg_embs=torch.transpose(msg_embs,0,1)
        #msg_embs+=self.positions
        #memory=encoder_output
        hx = self.sent_encoder(msg_embs)
        hx = torch.mean(hx,dim=0)
        #hx = torch.reshape(hx, (msg_shape[0], msg_shape[1], Param.room_emb_size))
        #hx = self.history_encoder(hx)        
        #hx=self.emb2idx(hx)
        return hx

    def forward(self,  message1,message2,choose_room_method="sample",  obj_nums=None):
        #shape从(a,b)到(a,1,b)
        #if len(message.shape) == 2: message = message.unsqueeze(1)
        type_prob = [];type_idx = []
        hx1 = self._encode_message(message1)
        hx2 = self._encode_message(message2)
        res = []
        res.append(self.fc_type(hx1))
        res.append(self.fc_color(hx2))
        for i in range(2):
            scores = self.softmax(res[i])        
            if choose_room_method == "greedy":
                type_idx.append(torch.argmax(scores, dim=1))
            elif choose_room_method == "sample":
                room_sampler = torch.distributions.Categorical(scores)
                idx=room_sampler.sample()
                # record actions
                type_prob.append(room_sampler.log_prob(idx))
                type_idx.append(idx)
        return type_idx, type_prob


    def cal_loss(self, type_prob, reward):
        loss = -sum([type_prob[i] * reward[i] for i in range(len(type_prob))])
        return loss
 
class AgentA(nn.Module):
    def __init__(self, lang_understand=None, lang_generate=None):
        super(AgentA, self).__init__()
        self.lang_generate = ELG_A() if lang_generate is None else lang_generate

    def describe_room(self, cur_room_emb, max_length, choose_token_method="sample"):
        return self.lang_generate(cur_room_emb, max_length, choose_token_method)

    def cal_guess_type_loss(self, spoken_token_prob, reward):
        return self.lang_generate.cal_loss(spoken_token_prob, reward)

class AgentB(nn.Module):
    def __init__(self, lang_understand=None, lang_generate=None, is_cal_all_route_init=False):
        super(AgentB, self).__init__()
        self.lang_understand = ELU_B() if lang_understand is None else lang_understand
        
    def guess_type(self,  message1,message2,choose_room_method="sample",guess_attribute="type"):
        return self.lang_understand( message1,message2,choose_room_method,guess_attribute)

    def cal_guess_type_loss(self, type_prob, reward):
        return self.lang_understand.cal_loss(type_prob, reward)

