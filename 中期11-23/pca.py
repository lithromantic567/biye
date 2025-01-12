from sklearn.model_selection import train_test_split
from SpotDiff1 import SpotDiff
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from Agents_new import *
from Dataset import EnvDataset_d,EnvDataset_all
from Param import *
import random
from sklearn.metrics import confusion_matrix
#from gr_fd import gr_fd
#from gr_fd_fc import *
from arg_parser import parse_arguments
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter

def encode_lang_vectors(lang_vectors, voc_embedding):
    """
    使用嵌入矩阵将符号序列编码为固定长度的向量。
    """
    device = voc_embedding.weight.device
    lang_emb = voc_embedding(lang_vectors.to(device))  # (batch_size, seq_len, embedding_dim)
    lang_encoded = lang_emb.mean(dim=1).detach().cpu().numpy()
 # 平均池化获得固定大小向量
    return lang_encoded
def find_dominant_label(cluster_labels, true_labels):
    """
    根据标注找到每个簇的主要标签。
    """
    dominant_labels = {}
    for cluster_id in np.unique(cluster_labels):
        indices = np.where(cluster_labels == cluster_id)[0]
        true_labels_for_cluster = true_labels[indices]
        most_common_label = Counter(true_labels_for_cluster).most_common(1)[0][0]
        dominant_labels[cluster_id] = most_common_label
    return dominant_labels
def analyze_relationship(model, data_loader, lang_voc_embedding):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    obs_vectors, lang_vectors, true_labels= [], [], []

    with torch.no_grad():
       for data,data_d,label in data_loader:
            cur_obs_info = data.to(torch.float32).to(device)  # 假设数据的第一个元素是观测向量
            cur_obs_d_info = data_d.to(torch.float32).to(device) 
            #diff_obs_info = (cur_obs_info - cur_obs_d_info).cpu().numpy()
            true_labels.append(label)
            type_idxes, token_probs,type_probs , sent = model(cur_obs_info,cur_obs_d_info,  choose_method="greedy", guess_attribute="type")
            obs_vectors.append((cur_obs_info).cpu().numpy())
            #obs_vectors.append(diff_obs_info)
            lang_vectors.append(sent.cpu())  # 假设 sent1 是符号序列

    obs_vectors = np.concatenate(obs_vectors, axis=0)  # (total_samples, 8, 8, 3)
    lang_vectors = torch.cat(lang_vectors, dim=0)  # (total_samples, seq_len)
    true_labels=np.concatenate(true_labels,axis=0)
    # 将语言向量编码为固定大小
    lang_encoded = encode_lang_vectors(lang_vectors, lang_voc_embedding)

    # 展平观测向量
    obs_flattened = obs_vectors.reshape(obs_vectors.shape[0], -1)  # (total_samples, 8*8*3)

    # PCA降维
    pca_obs = PCA(n_components=2)
    pca_lang = PCA(n_components=2)
    obs_2d = pca_obs.fit_transform(obs_flattened)
    lang_2d = pca_lang.fit_transform(lang_encoded)

    # 聚类
    kmeans_obs = KMeans(n_clusters=3, random_state=42)  # 假设5个簇
    kmeans_lang = KMeans(n_clusters=3, random_state=42)
    obs_clusters = kmeans_obs.fit_predict(obs_2d)
    lang_clusters = kmeans_lang.fit_predict(lang_2d)

    ## 找到每个簇的主要标签
    obs_dominant_labels = find_dominant_label(obs_clusters, true_labels)
    lang_dominant_labels = find_dominant_label(lang_clusters, true_labels)

    # 确保每个标签都有对应的映射
    all_labels = set(obs_dominant_labels.values()) | set(lang_dominant_labels.values())
    label_to_obs_center = {label: kmeans_obs.cluster_centers_[cluster] for cluster, label in obs_dominant_labels.items()}
    label_to_lang_center = {label: kmeans_lang.cluster_centers_[cluster] for cluster, label in lang_dominant_labels.items()}

    # 可视化
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 上层：obs_vectors
    ax.scatter(obs_2d[:, 0], obs_2d[:, 1], zs=1, c=obs_clusters, cmap='viridis', alpha=0.7, label="Obs Vectors")
    for center in kmeans_obs.cluster_centers_:
        ax.scatter(center[0], center[1], zs=1, color='red', marker='x', s=100)

    # 下层：lang_vectors
    ax.scatter(lang_2d[:, 0], lang_2d[:, 1], zs=0, c=lang_clusters, cmap='coolwarm', alpha=0.7, label="Lang Vectors")
    for center in kmeans_lang.cluster_centers_:
        ax.scatter(center[0], center[1], zs=0, color='blue', marker='x', s=100)
   
    # 连接匹配的簇中心
    for label in all_labels:
        if label in label_to_obs_center and label in label_to_lang_center:
            obs_center = label_to_obs_center[label]
            lang_center = label_to_lang_center[label]
            ax.plot([obs_center[0], lang_center[0]],
                    [obs_center[1], lang_center[1]],
                    [1, 0], 'k--')
    # 打印所有映射的标签和中心
    for label in all_labels:
        print(f"Label: {label}")
        if label in label_to_obs_center:
            print(f"  Observation Center: {label_to_obs_center[label]}")
        else:
            print("  Missing in Observation Clusters!")
        if label in label_to_lang_center:
            print(f"  Language Center: {label_to_lang_center[label]}")
        else:
            print("  Missing in Language Clusters!")


    # 设置图例和标签
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_zlabel("Layer (1=Obs, 0=Lang)")
    ax.legend()
    plt.title("3D Mapping between Obs Vectors and Lang Vectors with Labels")
    plt.show()
    plt.savefig("pca_col.png")
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
if __name__ == "__main__":
    setup_seed(args.seed)
    dataset=EnvDataset_d("Data/mt2/data")
    train_dataset, X_temp,= train_test_split(dataset, test_size=0.2)
    eval_dataset, test_dataset = train_test_split(X_temp, test_size=0.5)
    #guess_room_train(train_dataset,eval_dataset)
    #task_test(test_dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    model=SpotDiff().to(device)
    model.load_state_dict(torch.load("mt2/model.pth"))
    model.eval()      
    # 使用模型和数据集调用
    # 假设 model 是 MulTaskModel，data_loader 是 DataLoader 实例
    # 假设 lang_voc_embedding 是 AgentA 中 ELG_A 的 voc_embedding
    lang_voc_embedding = model.agentA.lang_generate.voc_embedding
    analyze_relationship(model, data_loader, lang_voc_embedding)
