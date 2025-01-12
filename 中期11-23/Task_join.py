from Agents_join import *
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
        
        type_feature,color_feature = self.room_embedding_A(cur_obs_info)
        # 类型任务的符号序列生成与理解
        type_sent, type_token_probs = self.agentA.describe_type(type_feature, Param.max_sent_len, choose_method)
        type_idx, type_prob = self.agentB.guess_type(type_sent, choose_method)

        # 颜色任务的符号序列生成与理解
        color_sent, color_token_probs = self.agentA.describe_color(color_feature, Param.max_sent_len, choose_method)
        color_idx, color_prob = self.agentB.guess_color(color_sent, choose_method)

        return type_idx, color_idx, (type_token_probs, color_token_probs), type_prob, color_prob, type_sent, color_sent

    def backward(self, token_probs, type_probs, color_probs, rewards_type,rewards_color):
        # 分别计算类型和颜色的损失
        loss_type = self.agentA.cal_type_loss(token_probs[0], rewards_type) + self.agentB.cal_type_loss(type_probs, rewards_type)
        loss_color = self.agentA.cal_color_loss(token_probs[1], rewards_color) + self.agentB.cal_color_loss(color_probs, rewards_color)

        # 联合优化
        total_loss = loss_type + loss_color
        total_loss.backward()
        return loss_type, loss_color