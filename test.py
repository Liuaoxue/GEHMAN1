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
d_node=128                     # Node feature dimension
epoch=args.epochs              # Number of training epochs
K=args.multihead                # Number of attention heads
lambda_1=args.lambda_1
lambda_2=args.lambda_2
lambda_3=args.lambda_3
device = torch.device('cuda:' + args.cuda if torch.cuda.is_available() else 'cpu')  # Set device
if __name__ == '__main__':
    #  Load the data and construct the graph
    g,friend_list_index_test,friend_list_index_val=data(d_node,"JK")       # Load graph and index lists
    g = g.to(device)                                                        # Move graph to the specified device

    # Define relationships in the graph
    rel_names = ['friend', 'visit', 'co_occurrence', 'live_with', 're_live_with', 'class_same', 're_visit',
                 'Active_association', 'Co_visiting']

    # Initialize the model
    model = Model(d_node, 256, 512, rel_names, K).to(device)

    #  Load node and edge features
    user_feats = g.nodes['user'].data['u_fe'].to(device)
    poi_feats = g.nodes['poi'].data['p_fe'].to(device)
    node_features = {'user': user_feats, 'poi': poi_feats}


    #  Load edge features for each relationship type
    friend_feats = g.edges['friend'].data['f_fe'].to(device)
    visit_feats = g.edges['visit'].data['v_fe'].to(device)
    co_occurrence_feat = g.edges['co_occurrence'].data['c_fe'].to(device)
    live_with_feats = g.edges['live_with'].data['l_fe'].to(device)
    re_live_with_feats = g.edges['re_live_with'].data['rl_fe'].to(device)
    class_same_feats = g.edges['class_same'].data['cl_fe'].to(device)
    re_visit_feats = g.edges['re_visit'].data['r_fe'].to(device)
    Active_association_feats = g.edges['Active_association'].data['Ac_fe'].to(device)
    Co_visiting_feats = g.edges['Co_visiting'].data['Co_fe'].to(device)

    # Combine edge attributes
    edge_attr = {'friend': friend_feats, 'visit': visit_feats, 'co_occurrence': co_occurrence_feat,
                 'live_with': live_with_feats, 're_live_with': re_live_with_feats, 'class_same': class_same_feats,
                 're_visit': re_visit_feats, 'Active_association': Active_association_feats,
                 'Co_visiting': Co_visiting_feats}

    #  Load the best model
    model.load_state_dict(torch.load("pth/best_model.pth"))
    model.eval()

   #  Construct negative graphs for contrastive learning
    negative_graph = construct_negative_graph(g, 5, ('user', 'friend', 'user')).to(device)

    # 3. Calculating user embeddings
    with torch.no_grad():
        pos_score, neg_score, node_emb, contrastive_loss = model(g, negative_graph, node_features, edge_attr,
                                                                 ('user', 'friend', 'user'))
        user_emb = node_emb['user']
    pos_edge_index_2 = []
    pos_edge_index = g.edges(etype=('user', 'friend', 'user'))                     # Get positive edge indices
    pos_edge_index_2.append(pos_edge_index[0].cpu().detach().numpy())
    pos_edge_index_2.append(pos_edge_index[1].cpu().detach().numpy())
    pos_edge_index_2 = torch.tensor(np.array(pos_edge_index_2)).to(device)
    
    # 4. Evaluate on the test set
    neg_edge_index = neg_edge_in(g, 5, ('user', 'friend', 'user'))                 # Generate negative edge indices
    link_labels = get_link_labels(pos_edge_index_2, neg_edge_index).to(device)     # Get link labels
    link_logits = model.predict(user_emb, pos_edge_index_2, neg_edge_index)        # Predict link likelihoods
    loss_cor = F.binary_cross_entropy_with_logits(link_logits, link_labels)        #  Compute loss (optional)

    #Calculating evaluation metrics
    test_auc, ap, top_k, f1_pos, f1_neg, f1_macro = test(user_emb, g, friend_list_index_test)

    #Printing Results
    print("Test AUC:", test_auc)
    print("Test AP:", ap)
    print("Top K Results:", top_k)
    print("F1 Score (Positive Class):", f1_pos)
    print("F1 Score (Negative Class):", f1_neg)
    print("F1 Score (Macro Average):", f1_macro)

    # Clean up the memory
    del negative_graph           # Delete negative graph to free memory
    torch.cuda.empty_cache()     # Clear CUDA memory


