import dgl
import numpy as np
import torch
from torch import nn
import torch as th
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import random
import tqdm
import sklearn.metrics
from torch import cosine_similarity
from dataset import *
from model import *
from utils import *
#Hyper-parameters

d_node=128
epoch=args.epochs
K=args.multihead     #multi_head
lambda_1=args.lambda_1
lambda_2=args.lambda_2
lambda_3=args.lambda_3
device = torch.device('cuda:' + args.cuda if torch.cuda.is_available() else 'cpu')
if __name__ == '__main__':
    g,friend_list_index_test,friend_list_index_val=data(d_node,"JK")
    g = g.to(device)
    # user_emb=torch.tensor(np.load("data/save_user_embedding/best_auc_JK0.8760630365484157.npy")).to(device)
    # test_auc, ap,f1_pos, f1_neg, f1_macro = test(user_emb, g, friend_list_index_test)
    # print("test_auc:",test_auc)
    # print("test_ap:",ap)
    rel_names = ['friend', 'visit', 'co_occurrence', 'live_with', 're_live_with', 'class_same', 're_visit',
                 'Active_association', 'Co_visiting']
    model = Model(d_node, 256, 512, rel_names, K).to(device)
    user_feats = g.nodes['user'].data['u_fe'].to(device)
    poi_feats = g.nodes['poi'].data['p_fe'].to(device)
    node_features = {'user': user_feats, 'poi': poi_feats}
    friend_feats = g.edges['friend'].data['f_fe'].to(device)
    visit_feats = g.edges['visit'].data['v_fe'].to(device)
    co_occurrence_feat = g.edges['co_occurrence'].data['c_fe'].to(device)
    live_with_feats = g.edges['live_with'].data['l_fe'].to(device)
    re_live_with_feats = g.edges['re_live_with'].data['rl_fe'].to(device)
    class_same_feats = g.edges['class_same'].data['cl_fe'].to(device)
    re_visit_feats = g.edges['re_visit'].data['r_fe'].to(device)
    Active_association_feats = g.edges['Active_association'].data['Ac_fe'].to(device)
    Co_visiting_feats = g.edges['Co_visiting'].data['Co_fe'].to(device)
    edge_attr = {'friend': friend_feats, 'visit': visit_feats, 'co_occurrence': co_occurrence_feat,
                 'live_with': live_with_feats, 're_live_with': re_live_with_feats, 'class_same': class_same_feats,
                 're_visit': re_visit_feats, 'Active_association': Active_association_feats,
                 'Co_visiting': Co_visiting_feats}

    # 1. 加载最佳模型
    model.load_state_dict(torch.load("pth/best_model.pth"))
    model.eval()

    # 2. 构建负图
    negative_graph = construct_negative_graph(g, 5, ('user', 'friend', 'user')).to(device)

    # 3. 计算用户嵌入
    with torch.no_grad():
        pos_score, neg_score, node_emb, contrastive_loss = model(g, negative_graph, node_features, edge_attr,
                                                                 ('user', 'friend', 'user'))
        user_emb = node_emb['user']
    pos_edge_index_2 = []
    pos_edge_index = g.edges(etype=('user', 'friend', 'user'))
    pos_edge_index_2.append(pos_edge_index[0].cpu().detach().numpy())
    pos_edge_index_2.append(pos_edge_index[1].cpu().detach().numpy())
    pos_edge_index_2 = torch.tensor(np.array(pos_edge_index_2)).to(device)
    # 4. 在测试集上进行评估
    neg_edge_index = neg_edge_in(g, 5, ('user', 'friend', 'user'))  # 生成负边索引
    link_labels = get_link_labels(pos_edge_index_2, neg_edge_index).to(device)  # 获取链接标签
    link_logits = model.predict(user_emb, pos_edge_index_2, neg_edge_index)  # 预测链接
    loss_cor = F.binary_cross_entropy_with_logits(link_logits, link_labels)  # 计算损失（可选）

    # 计算评估指标
    test_auc, ap, top_k, f1_pos, f1_neg, f1_macro = test(user_emb, g, friend_list_index_test)

    # 打印结果
    print("Test AUC:", test_auc)
    print("Test AP:", ap)
    print("Top K Results:", top_k)
    print("F1 Score (Positive Class):", f1_pos)
    print("F1 Score (Negative Class):", f1_neg)
    print("F1 Score (Macro Average):", f1_macro)

    # 清理内存
    del negative_graph
    torch.cuda.empty_cache()


