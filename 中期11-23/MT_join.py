# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
from Task_join import MulTaskModel
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from Agents_join import *
from Dataset import EnvDataset_obj_col
from Param import *
import random
from sklearn.metrics import confusion_matrix
#from gr_fd import gr_fd
#from gr_fd_fc import *
from arg_parser import parse_arguments
#用同一个数据同时猜删掉的物体类型和颜色

args = parse_arguments()

def clear_file(file_path):
    with open(file_path+"/train_acc.txt","w") as f:
        f.write('')
    with open(file_path+"/train_reward.txt","w") as f:
        f.write('')
    with open(file_path+"/train_loss_A.txt","w") as f:
        f.write('')
    with open(file_path+"/train_loss_B.txt","w") as f:
        f.write('')
    with open(file_path+"/eval_acc.txt","w") as f:
        f.write('')
    with open(file_path+"/pattern.txt","w") as f:
        f.write('')

def process_result(i,tgt,pred_type,total_reward,total_loss_type,save_path):
    tgt = np.concatenate(tgt, axis=0)
    pred = np.concatenate(pred_type, axis=0)
    print("|",end='',flush=True)
    #不用10个epoch输出一次，因为很早就收敛了，如果10个epoch输出一次波动很大
    
    acc_train=np.mean(tgt == pred)
    with open(save_path+"/train_acc.txt",'a') as fp:
        fp.write(str(acc_train)+'\n')
    with open(save_path+"/train_reward.txt",'a') as fp:
        fp.write(str(total_reward)+'\n')
    with open(save_path+"/train_loss.txt",'a') as fp:
        fp.write(str(total_loss_type)+'\n')  
    
    print()
    print("epoch{}: \nacc = {}, loss A = {}, loss B = {}, reward={}".format(i, acc_train, total_loss_type,total_reward),flush=True) 
    #print("epoch{}: \nacc = {}, loss A = {}, loss B = {}".format(i, np.mean(accum_tgt == accum_pred), total_loss_A, total_loss_B))           
    

def guess_room_train(train_dataset,eval_dataset):
    clear_file("Mul_join/Task1")
    clear_file("Mul_join/Task2")
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    """
    train
    :return:
    """
    #train_keyset = EnvDataset_all(Param.key_train_dir)
    #train_keyloader = DataLoader(train_keyset, batch_size=args.batch_size)
    #train_ballset = EnvDataset_all(Param.ball_train_dir)
    #train_ballloader = DataLoader(train_ballset, batch_size=args.batch_size)
    #train_boxset = EnvDataset_all(Param.box_train_dir)
    #train_boxloader = DataLoader(train_boxset, batch_size=args.batch_size)
    #type_obs_info=np.concatenate((keyset, ballset, boxset))
    #print(type_obs_info)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    
    #task = gr_fd().to(device)
    task = MulTaskModel().to(device)
    # if Param.is_gpu: task = task.to(Param.gpu_device)
    opt = Adam(task.parameters(), lr=Param.lr_task, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4)
    # opt = SGD(task.parameters(), lr=Param.lr, momentum=0.9)
    best_eval1=0; best_eval2=0
    for i in range(1000):
        tgt1 = []; tgt2=[]
        pred_type= []; pred_color=[]
        cur_sent = None
        total_loss_A1 = 0; total_loss_B1 = 0
        total_loss_A2 = 0; total_loss_B2 = 0
        total_reward_type=0; total_reward_color=0
        total_loss_type =0;total_loss_color=0
        
        #for  data_batch,key,ball,box in zip(train_dataloader,train_keyloader,train_ballloader,train_boxloader):
        for input in train_dataloader: 
            opt.zero_grad()
            #data,data_d,label=data_batch
            #data,data_d,label1,label2=input
            data,label1,label2=input

            tgt1.append(label1)
            tgt2.append(label2)
            #data,data_d, label,key,ball,box = data.to(device), data_d.to(device),label.to(device),key.to(device),ball.to(device),box.to(device)
    
            cur_obs_info = data.to(torch.float32).to(device)
            #cur_obs_d_info= data_d.to(torch.float32).to(device)
            
            #key=key.to(torch.float32)
            #ball=ball.to(torch.float32)
            #box=box.to(torch.float32)
            #print(step)
            task.train()
            task.agentA.train()
            task.agentB.train()
            # --- FORWARD ----
            # num_room = num_room.to(torcht.float32); num_obs = num_obs.to(torch.float32)
            # cur_env_info = cur_env_info.to(torch.float32)
            
            #type_idxes, token_probs,type_probs , sent = task(cur_obs_info,cur_obs_d_info,key, ball, box)
            #type_idxes, token_probs1,token_probs2,type_probs ,sent1,sent2 = task(cur_obs_info,cur_obs_d_info)
            #type_idxes, token_probs1,token_probs2,type_probs ,sent1,sent2 = task(cur_obs_info)
            type_idxes, color_idxes, token_probs, type_probs, color_probs, sent_type, sent_color = task(cur_obs_info)

            
            
            pred_type.append(type_idxes.cpu().numpy())
            # --- BACKWARD ---
            reward_type = np.ones_like(type_idxes.cpu().numpy())
            reward_type[type_idxes.cpu().numpy() != label1.cpu().numpy() ] = -1
            pred_color.append(color_idxes.cpu().numpy())
            # --- BACKWARD ---
            reward_color = np.ones_like(color_idxes.cpu().numpy())
            reward_color[color_idxes.cpu().numpy() != label2.cpu().numpy() ] = -1
            #reward *= Param.reward
            #positive_numbers = [x for x in reward if x > 0]
            #print(positive_numbers)
            #cur_loss_A1, cur_loss_B1, cur_loss_A2, cur_loss_B2 = task.backward(token_probs1, type_probs[0], torch.Tensor(reward_type1).to(device),token_probs2, type_probs[1], torch.Tensor(reward_type2).to(device))
            loss_type, loss_color = task.backward(token_probs, type_probs, color_probs, torch.Tensor(reward_type).to(device), torch.Tensor(reward_color).to(device))
            total_loss_type+=loss_type.item();total_loss_color+=loss_color.item()
            #total_loss_A1 += cur_loss_A1.item(); total_loss_B1 += cur_loss_B1.item()
            #total_loss_A2 += cur_loss_A2.item(); total_loss_B2 += cur_loss_B2.item()
            total_reward_type+=sum(reward_type)
            total_reward_color+=sum(reward_color)
            
            opt.step()
        
        process_result(i,tgt1,pred_type,total_reward_type,total_loss_type,"Mul_join/Task1")
        process_result(i,tgt2,pred_color,total_reward_color,total_loss_color,"Mul_join/Task2")
        
        #task.eval()
        #task.agentA.eval()
        #task.agentB.eval()
        acc_eval1,acc_eval2=guess_room_evaluate(eval_dataset, task,save_path="Mul_join/")
        
        if acc_eval1>best_eval1:
            best_eval1=acc_eval1
            torch.save(task.state_dict(), "Mul_join/Task1/model.pth")
        if acc_eval2>best_eval2:
            best_eval2=acc_eval2
            torch.save(task.state_dict(), "Mul_join/Task2/model.pth")


def guess_room_evaluate(eval_dataset,model,save_path):
    """
    evaluation
    :param model:
    :return:
    """
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    #eval_keyset = EnvDataset_all(Param.key_eval_dir)
    #eval_keyloader = DataLoader(eval_keyset, batch_size=args.batch_size)
    #eval_ballset = EnvDataset_all(Param.ball_eval_dir)
    #eval_ballloader = DataLoader(eval_ballset, batch_size=args.batch_size)
    #eval_boxset = EnvDataset_all(Param.box_eval_dir)
    #eval_boxloader = DataLoader(eval_boxset, batch_size=args.batch_size)
    
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)
    model.to(device).eval()
    
    tgt1 = []; pred1 = []
    tgt2= []; pred2=[]
    #total_loss=0
    with torch.no_grad():
        
        acc_eval1=[];acc_eval2=[]
        #for data_batch,key,ball,box in zip(eval_dataloader,eval_keyloader,eval_ballloader,eval_boxloader): 
        for data,label1,label2 in eval_dataloader:
            #data,data_d,label=data_batch
            tgt1.append(label1)
            tgt2.append(label2)
            #data,data_d, label,key,ball,box = data.to(device), data_d.to(device), label.to(device),key.to(device),ball.to(device),box.to(device)
            cur_obs_info = data.to(torch.float32).to(device)
            #cur_obs_d_info = data_d.to(torch.float32).to(device)
            #key=key.to(torch.float32)
            #ball=ball.to(torch.float32)
            #box=box.to(torch.float32)
            #type_idxes,token_probs1,token_probs2,type_probs, sent1,sent2 = model(cur_obs_info,cur_obs_d_info,  choose_method="greedy")
            type_idxes, color_idxes, token_probs, type_probs, color_probs, sent_type, sent_color = model(cur_obs_info)

            for i in range(5,10):
                print(sent_type[i],sent_color[i])
                print(type_idxes[i],color_idxes[i])
                print(label1[i],label2[i])
                print("-----")
            pred1.append(type_idxes.cpu().numpy())
            pred2.append(color_idxes.cpu().numpy())
        tgt1 = np.concatenate(tgt1, axis=0)
        pred1 = np.concatenate(pred1, axis=0)
        acc_eval1=np.mean(tgt1 == pred1)
        tgt2 = np.concatenate(tgt2, axis=0)
        pred2 = np.concatenate(pred2, axis=0)
        acc_eval2=np.mean(tgt2 == pred2)
        #total_loss=total_loss/(len(eval_dataset)/args.batch_size)
        with open(save_path+"eval_acc1.txt",'a') as f:
            f.write(str(acc_eval1)+'\n')
        with open(save_path+"eval_acc2.txt",'a') as f:
            f.write(str(acc_eval2)+'\n')
        #with open("fd_b/eval_loss.txt",'a') as f:
            #f.write(str(total_loss)+'\n')
        print("eval acc 1= {},acc 2={}".format(acc_eval1,acc_eval2))
        return acc_eval1,acc_eval2

def pro_pattern(tgt,pred,sents,guess_attribute,save_path):
    s={}
    d=[]
    step=0
    for i in sents:
        i=tuple(i)  
        d.append(i)      
        if i in s:
            s[i]+=1
        else:
            s[i]=1
    
    for message in s:
        if guess_attribute=="type":
            label1=[0,0,0];label2=[0,0,0]
        else:
            label1=[0,0,0,0,0,0];label2=[0,0,0,0,0,0]
        for i in range(len(d)):
            if d[i]==message:
                label1[tgt[i]]+=1
                label2[pred[i]]+=1
        with open(save_path+"/pattern.txt",'a') as f:
            f.write(str(message)+':'+str(label1)+str(label2)+'\n')
    conf_label12 = confusion_matrix(tgt, pred)
    print(conf_label12)
    
    acc_test=np.mean(tgt == pred)
    print("test acc = {}".format(acc_test))
def task_test(test_dataset,save_path):
    """
    evaluation
    :param model:
    :return:
    """
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    model=MulTaskModel().to(device)
    model.load_state_dict(torch.load(save_path+"/Task1/model.pth"))
    model.eval()
    
    tgt1 = []; pred1 = []; sents1=[]
    tgt2 = []; pred2 = []; sents2=[]
    #total_loss=0
    with torch.no_grad():
        
        acc_test=[]
        for data,label1,label2 in test_dataloader:
            tgt1.append(label1)
            tgt2.append(label2)
            cur_obs_info = data.to(torch.float32).to(device)
            #cur_obs_d_info = data_d.to(torch.float32).to(device)
            type_idxes,token_probs1,token_probs2,type_probs, sent1,sent2 = model(cur_obs_info,  choose_method="greedy")
           
            pred1.append(type_idxes[0].cpu().numpy())
            sents1.append(sent1.cpu().numpy())
            pred2.append(type_idxes[1].cpu().numpy())
            sents2.append(sent2.cpu().numpy())
            
        tgt1 = np.concatenate(tgt1, axis=0)
        pred1 = np.concatenate(pred1, axis=0)
        sents1 = np.concatenate(sents1, axis=0)
        tgt2 = np.concatenate(tgt2, axis=0)
        pred2 = np.concatenate(pred2, axis=0)
        sents2 = np.concatenate(sents2, axis=0)
        pro_pattern(tgt1,pred1,sents1,"type",save_path+"/Task1")
        pro_pattern(tgt2,pred2,sents2,"color",save_path+"/Task2")
        
        
        return acc_test

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    setup_seed(args.seed)

    #dataset=EnvDataset_d("../Data/Env_fd_3/data")
    dataset=EnvDataset_obj_col("Data/mmoe_pre/data")
    train_dataset, X_temp,= train_test_split(dataset, test_size=0.2)
    eval_dataset, test_dataset = train_test_split(X_temp, test_size=0.5)


    guess_room_train(train_dataset,eval_dataset)
    #task_test(test_dataset,"Mul_mmoe")
    

