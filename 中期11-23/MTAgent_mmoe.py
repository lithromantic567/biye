from Agents_mmoe import *
#from ceiling import *
from Param import *
from mt_pre import *
# 采用mmoe的预训练
def safe_grad_norm(parameters):
        grads = [p.grad for p in parameters if p.grad is not None]
        grads = [g for g in grads if not (torch.isnan(g).any() or torch.isinf(g).any())]
        if len(grads) == 0:  # 如果所有梯度都无效，返回0
            return torch.tensor(0.0, device=parameters[0].device)
        return torch.norm(torch.cat([g.view(-1) for g in grads]))
class MulTaskModel(nn.Module):
    def __init__(self, agentA=None, agentB=None):
        super(MulTaskModel, self).__init__()
        self.agentA = AgentA() if agentA is None else agentA
        self.agentB = AgentB() if agentB is None else agentB
        
        # 创建ConvA和ConvB模型实例
        #self.room_embedding_A = ConvNet()
        #self.room_embedding_B = ConvNet()
        self.room_embedding_A = CustomMMoE()
        #self.room_embedding_B = CustomMMoE()
        
        
        state_dict = torch.load('mmoe_pre/model.pth')
        params = {k.replace('task_specific_features.', ''): v for k, v in state_dict.items() if k.startswith('task_specific_features')}
        # 加载预训练的模型参数
        self.room_embedding_A.load_state_dict(params)
        #self.room_embedding_B.load_state_dict(params)
        # 将模型设置为评估模式
        #self.room_embedding_A.eval()
        #self.room_embedding_B.eval()
        #self.room_embedding_A.requires_grad=False
        #self.room_embedding_B.requires_grad=False
        
        self.diff_fc=nn.Sequential(
            nn.Linear(Param.room_emb_size*2, Param.room_emb_size),
            nn.ReLU()
        )
        # 获取设备
        # 初始化任务权重并对齐设备
        self.task_weights = nn.Parameter(torch.ones(2))  # 初始化为1，并放置在设备上
        self.alpha = 0.12
            
        
        #self.room_embedding_A = GridEmbedding()  # initial state
        #self.room_embedding_B = GridEmbedding()
        # self.room_embedding_B = self.room_embedding_A  # share

    def forward(self, cur_obs_info,  choose_method="sample", history_sents=None, env_ids=None, route_len=None):
        #batch_size*1 50
        #tgt_types_arr = np.array(label_type).astype(int)
        #tgt_colors_arr = np.array(label_color).astype(int)
        #(50,9,3,3,3)
        #obs_info= env_info[:,:,2:5,2:5,:]
        #(50,9,50)->(50,50)
        
        room_embs_A1,room_embs_A2 = self.room_embedding_A(cur_obs_info)
        #room_d_embs_A1,room_d_embs_A2 = self.room_embedding_A(cur_obs_d_info)
        #x = torch.cat([room_embs_A,room_d_embs_A],dim=-1).squeeze()#(20,100)    
        #print(room_embs_A)
        #print(room_d_embs_A)
        #diff_emb=self.diff_fc(x).squeeze()
        #diff_emb=(room_embs_A-room_d_embs_A).squeeze().detach()
        ##diff_emb1=(room_embs_A1-room_d_embs_A1).squeeze()
        ##diff_emb2=(room_embs_A2-room_d_embs_A2).squeeze()
        #diff_emb=x
        #print(diff_emb)
        #sent1, token_probs1 = self.agentA.describe_room(diff_emb1, Param.max_sent_len, choose_method)
        
        #sent2, token_probs2 = self.agentA.describe_room(diff_emb2, Param.max_sent_len, choose_method)
        sent1, token_probs1 = self.agentA.describe_room(room_embs_A1, Param.max_sent_len, choose_method)
        
        sent2, token_probs2 = self.agentA.describe_room(room_embs_A2, Param.max_sent_len, choose_method)
        
        type_idx, type_prob = self.agentB.guess_type(sent1,sent2, choose_method)   
        
        #color_idx, color_prob = self.agentB.guess_color(sent, choose_method)
        return type_idx, token_probs1,token_probs2,type_prob , sent1,sent2
    '''
    def backward(self, token_probs1, type_prob1, reward1,token_probs2,type_prob2,reward2):
        lossA1 = self.agentA.cal_guess_type_loss(token_probs1, reward1)
        lossB1 = self.agentB.cal_guess_type_loss(type_prob1, reward1)
        lossA2 = self.agentA.cal_guess_type_loss(token_probs2, reward2)
        lossB2 = self.agentB.cal_guess_type_loss(type_prob2, reward2)
        lossA=lossA1+lossA2
        lossB=lossB1+lossB2
        lossA.backward()
        lossB.backward()
        return lossA1, lossB1,lossA2,lossB2
    '''


    def backward(self, token_probs1, type_prob1, reward1, token_probs2, type_prob2, reward2):
        # 计算每个任务的损失
        lossA1 = self.agentA.cal_guess_type_loss(token_probs1, reward1)
        lossB1 = self.agentB.cal_guess_type_loss(type_prob1, reward1)
        lossA2 = self.agentA.cal_guess_type_loss(token_probs2, reward2)
        lossB2 = self.agentB.cal_guess_type_loss(type_prob2, reward2)

        # 总损失
        total_loss = lossA1 + lossB1 + lossA2 + lossB2
        total_loss.backward(retain_graph=True)

        # 安全计算梯度范数
        grad_norm_task1 = safe_grad_norm(self.agentA.parameters())
        grad_norm_task2 = safe_grad_norm(self.agentB.parameters())

        # 目标梯度
        mean_grad_norm = (grad_norm_task1 + grad_norm_task2) / 2 + 1e-8
        target_grad_norm_task1 = mean_grad_norm * (self.task_weights[0] / self.task_weights.mean())
        target_grad_norm_task2 = mean_grad_norm * (self.task_weights[1] / self.task_weights.mean())

        # 更新任务权重
        self.task_weights.data[0] -= self.alpha * (grad_norm_task1 - target_grad_norm_task1).clamp(min=-1, max=1)
        self.task_weights.data[1] -= self.alpha * (grad_norm_task2 - target_grad_norm_task2).clamp(min=-1, max=1)

        # 确保权重非负
        self.task_weights.data = torch.relu(self.task_weights.data)

        return lossA1, lossB1, lossA2, lossB2
