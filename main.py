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
import os
from config import *

# Set PyTorch CUDA memory configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.cuda.set_per_process_memory_fraction(0.8)
from torch.cuda.amp import GradScaler, autocast
torch.cuda.empty_cache()

# Parse command-line arguments
args=parse()
device = torch.device('cuda:' + args.cuda if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
city=args.city


#Hyper-parameters

d_node=128            # Dimension of node features
epoch=args.epochs          # Number of training epochs
K=args.multihead         # Number of heads for multi-head attention
lambda_1=args.lambda_1       # Weight for contrastive loss
lambda_2=args.lambda_2      # Weight for link prediction loss
lambda_3=args.lambda_3    # Weight for margin loss
file='output1/'+str(city)+'-*-_multi_head_'+str(K)+'lambda_1_'+str(lambda_1)+'lambda_2_'+str(lambda_2)+'lambda_3_------'+str(lambda_3)+'.txt'
print(file)
if __name__ == '__main__':
    # Load data
    g,friend_list_index_test,friend_list_index_val=data(d_node,city)     # Load graph and indices
    g = g.to(device)                                                         # Move graph to the specified device
    etype = g.etypes                                                          # Get edge types
    # rel_names = ['friend', 'visit', 'co_occurrence', 'live_with', 're_live_with', 'class_same', 're_visit']
    #rel_names = ['friend', 'visit', 'co_occurrence', 'live_with', 're_live_with', 'class_same', 're_visit']
    rel_names = ['friend', 'visit', 'co_occurrence', 'live_with', 're_live_with', 'class_same', 're_visit','Active_association','Co_visiting']
    
    # Initialize model
    model = Model(d_node, 256, 512, rel_names, K).to(device)
    user_feats = g.nodes['user'].data['u_fe'].to(device)
    poi_feats = g.nodes['poi'].data['p_fe'].to(device)

    # Dictionary of node features
    node_features = {'user': user_feats, 'poi': poi_feats}

    # Load edge features
    friend_feats = g.edges['friend'].data['f_fe'].to(device)
    visit_feats = g.edges['visit'].data['v_fe'].to(device)
    co_occurrence_feat = g.edges['co_occurrence'].data['c_fe'].to(device)
    live_with_feats = g.edges['live_with'].data['l_fe'].to(device)
    re_live_with_feats = g.edges['re_live_with'].data['rl_fe'].to(device)
    class_same_feats = g.edges['class_same'].data['cl_fe'].to(device)
    re_visit_feats = g.edges['re_visit'].data['r_fe'].to(device)
    Active_association_feats = g.edges['Active_association'].data['Ac_fe'].to(device)
    Co_visiting_feats = g.edges['Co_visiting'].data['Co_fe'].to(device)
    
    #Edge features dictionar
    edge_attr = {'friend': friend_feats, 'visit': visit_feats, 'co_occurrence': co_occurrence_feat,'live_with': live_with_feats, 're_live_with': re_live_with_feats, 'class_same': class_same_feats,'re_visit': re_visit_feats,'Active_association':Active_association_feats,'Co_visiting':Co_visiting_feats}

    # Get positive edge indices
    pos_edge_index_2 = []
    pos_edge_index = g.edges(etype=('user', 'friend', 'user'))             # Get positive edges
    pos_edge_index_2.append(pos_edge_index[0].cpu().detach().numpy())          # Append source nodes
    pos_edge_index_2.append(pos_edge_index[1].cpu().detach().numpy())        # Append target nodes
    pos_edge_index_2 = torch.tensor(np.array(pos_edge_index_2)).to(device)   # Convert to tensor

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)     # Adam optimizer
    best_auc = 0     # Best AUC tracker
    best_ap = 0     # Best AP tracker
    print(city)

    
    # Initialize gradient scaler
    scaler = GradScaler()

    
    for epoch in range(epoch):
        # Clear CUDA cache to manage memory
        torch.cuda.memory_reserved()  # Check reserved memory
        torch.cuda.empty_cache()  # Release unused memory

        # Construct negative graph
        negative_graph = construct_negative_graph(g, 5, ('user', 'friend', 'user'))
        pos_score, neg_score, node_emb, contrastive_loss = model(g, negative_graph, node_features, edge_attr, ('user', 'friend', 'user'))
        user_emb = node_emb['user']

        # Clean up temporary variables
        del negative_graph             # Remove negative graph to free memory
   

        # Get negative edge indices
        neg_edge_index = neg_edge_in(g, 5, ('user', 'friend', 'user'))                # Generate negative edges
        link_labels = get_link_labels(pos_edge_index_2, neg_edge_index).to(device)     # Create link labels
        link_logits = model.predict(user_emb, pos_edge_index_2, neg_edge_index)         # Predict link probabilities

        # Compute losses
        loss_cor = F.binary_cross_entropy_with_logits(link_logits, link_labels)      # Link prediction loss
        loss = margin_loss(pos_score, neg_score) * lambda_3 + loss_cor * lambda_2 + contrastive_loss*lambda_1     # Total loss
            
        # Backpropagation
        opt.zero_grad()   # Reset gradients
        loss.backward()   # Compute gradients
        opt.step()       # Update weights


        # Clean up intermediate variables
        del link_labels
        del link_logits
        del neg_edge_index

        
        # Release cache
        torch.cuda.empty_cache()

        # Log progress every 10 epochs
        if epoch % 10 == 0:
            print("epoch:", epoch)
            print("LOSS:", loss.item())
            # test_auc, ap,top_k = test(user_emb,g,friend_list_index_test)
            
            # Val the model and compute metrics
            test_auc, ap,top_k , f1_pos, f1_neg, f1_macro = test(user_emb, g, friend_list_index_val)
            if test_auc > best_auc:
                best_auc = test_auc            # Update best AUC
                print("best_auc:", best_auc)
                best_model_state = model.state_dict()      # Save model state
                torch.save(best_model_state, "pth/best_model.pth")   #Save best model 
                np.save("data/save_user_embedding/SP/sp_3/_best_auc_SP" + str(best_auc) + ".npy", user_emb.cpu().detach().numpy())
            if ap > best_ap:
                best_ap = ap            # Update best AP
                print("beat_ap:", ap)

    
            need_write = f"epoch {epoch} loss: {loss.item()} best_auc: {best_auc} best_ap: {best_ap} " \
                             f"f1_pos: {f1_pos} f1_neg: {f1_neg} f1_macro: {f1_macro}"

            #need_write="epoch"+str(epoch)+" best_auc: "+str(best_auc)+" best_ap: "+str(best_ap)
            top='top_1+'+str(top_k[0])+' top_5+'+str(top_k[1])+' top_10+'+str(top_k[2])+' top_15+'+str(top_k[3])+' top_20+'+str(top_k[4])
            with open(file, 'a+') as f:
                f.write(need_write + '\n')              # Write loss and metrics
                f.write(top + '\n')                     # Write top-k metrics
            
                
                

