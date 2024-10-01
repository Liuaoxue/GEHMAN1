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
from config import *
args=parse()

device = torch.device('cuda:' + args.cuda if torch.cuda.is_available() else 'cpu')

#Generate link labels
def get_link_labels(pos_edge_index, neg_edge_index):

    """
    Generate a tensor for link labels, with 1 for positive samples and 0 for negative samples.

    :param pos_edge_index: (Positive edge indices)
    :param neg_edge_index: (Negative edge indices)
    :return: (Tensor containing link labels)
    """
    # returns a tensor:
    # [1,1,1,1,...,0,0,0,0,0,..] with the number of ones is equel to the lenght of pos_edge_index
    E = pos_edge_index.size(1) + neg_edge_index.size(1)                                       # Calculate total number of edges
    link_labels = torch.zeros(E, dtype=torch.float, device=device)                            # Initialize label tensor
    link_labels[:pos_edge_index.size(1)] = 1                                                  # Set positive sample labels to 1
    return link_labels
    
#Constructing negative sample graph
def construct_negative_graph(graph, k, etype):
    
    """
    Construct a negative sample graph, generating non-connected edges.

    :param graph:  (Original graph)
    :param k: (Number of negative samples per edge)
    :param etype:  (Edge type)
    :return:  (Negative sample graph)
    """

    
    utype, _, vtype = etype                                                                  # Get source and target types of the edge
    src, dst = graph.edges(etype=etype)                                                      # Get source and target nodes of edges
    neg_src = src.repeat_interleave(k).to(device)                                            # Repeat source nodes to generate negative samples
    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,)).to(device)
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},                                                         # Construct negative sample graph
        num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})            # Specify number of nodes

class HeteroDotProductPredictor(nn.Module):

    """
    Calculate similarity scores between nodes of a specific edge type in a heterograph.
    """
    
    def forward(self, graph, h, etype):

        """
        Calculate similarity scores
        :param graph:  (Input graph)
        :param h:  (Node features)
        :param etype:  (Edge type)
        :return:  (Similarity scores)
        """
        
        utype, _, vtype = etype
        if utype==vtype:
            src, dst = graph.edges(etype=etype)                                             # source and target types of the edge
            h2=h[utype]                                                                     # Get node features
            logits = cosine_similarity(h2[src], h2[dst]) # Calculate cosine similarity
            logits_2 = torch.relu(logits)                                                   # Apply ReLU activation
            return logits_2
        if utype!=vtype:
            src, dst = graph.edges(etype=etype)                                             # Get source and target nodes
            h2_u=h[utype]                                                                   # Get source node features
            h2_v=h[vtype]                                                                   # Get target node features
            logits = cosine_similarity(h2_u[src], h2_v[dst])                                # Calculate cosine similarity
            logits_2 = torch.relu(logits)                                                   # Apply ReLU activation
            return logits_2


# Calculates a contrastive loss function to pull positive pairs closer and push negative pairs further apart.
def contrastive_loss(user_emb,g):
    
    """
    Calculate contrastive loss.
    :param user_emb:  (User embeddings)
    :param g:  (Graph)
    :return:  (Contrastive loss value)
    """   
    
    # adj_friend=g.adj(scipy_fmt='coo',etype='friend')
    adj_friend = g.adj_external(scipy_fmt='coo', etype='friend')                             # Get adjacency matrix for friendship relation
    adj_friend = adj_friend.todense()                                                        # Convert to dense matrix
    row, col = np.diag_indices_from(adj_friend)
    adj_friend[row, col] = 1                                                                 # Set self-connections to 1
    # a=torch.norm(user_emb[0],dim=-1,keepdim=True)
    user_emb_norm = torch.norm(user_emb, dim=-1, keepdim=True)                               # Calculate norm of user embeddings

    dot_numerator = torch.mm(user_emb, user_emb.t())                                         # Calculate dot product
    dot_denominator = torch.mm(user_emb_norm, user_emb_norm.t())                             # Calculate dot product of norms
    sim = torch.exp(dot_numerator / dot_denominator / 0.2)                                   # Calculate similarity
    x = (torch.sum(sim, dim=1).view(-1, 1) + 1e-8)                                           # Prevent division by zero
    matrix_mp2sc = sim / (torch.sum(sim, dim=1).view(-1, 1) + 1e-8)                          # Normalize similarity
    adj_friend = torch.tensor(adj_friend).to(device)                                         # Convert adjacency matrix to tensor
    lori_mp = -torch.log(matrix_mp2sc.mul(adj_friend).sum(dim=-1)).mean()                    # Calculate contrastive loss
    return lori_mp
    
#Calculates a margin loss to ensure positive pairs have higher similarity scores than negative pairs.
def margin_loss(pos_score, neg_score):
    """
    Calculate margin loss.
    :param pos_score:  (Positive similarity scores)
    :param neg_score:  (Negative similarity scores)
    :return:  (Margin loss value)
    """
    n_edges = pos_score.shape[0]                                                                 # Get number of edges
    return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()        #  Calculate and return loss
    
#Generates negative edges.
def neg_edge_in(graph,k,etype):
   
    """
    Generate negative edge indices.
    :param graph:  (Input graph)
    :param k:  (Number of negative samples per edge)
    :param etype:  (Edge type)
    :return:  (Negative edge indices)
    """
    
    # edgtypes= ('user', 'friend', 'user')
    utype, _, vtype = etype                                                                   # Get source and target types of the edge
    src, dst = graph.edges(etype=etype)                                                       # Get source and target nodes
    neg_src = src.repeat_interleave(k)                                                        # Repeat source nodes to generate negative samples
    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,)).to(device)            # Randomly generate target nodes
    neg_edge_ = torch.stack([neg_src, neg_dst], dim=0)                                        # Combine negative edge indices
    return neg_edge_




#Test link prediction performance

def test(user_emb,g,friend_list_index_test):
    """
    Test the performance of link prediction.
    :param user_emb:  (User embeddings)
    :param g:  (Input graph)
    :param friend_list_index_test:  (Test friendship list indices)
    :return: AUC, AP, top_k, F1 score
    """

    
    src, dst = g.edges(etype='friend')                                           # Get source and target of friendship edges
    src=list(src.cpu().detach().numpy())                                         # Convert to list
    dst=list(dst.cpu().detach().numpy())                                         # Convert to list
    friend_ture={}                                                               # Store true friendship relations
    for i in range(len(src)):
        if src[i] in friend_ture.keys():
            friend_ture[src[i]]=friend_ture[src[i]]+[dst[i]]
        else:
            friend_ture[src[i]]=[dst[i]]
    test_pos_src, test_pos_dst = friend_list_index_test[0], friend_list_index_test[1]     
    
    # Generate negative sample pairs
    seed = 30100
    torch.manual_seed(30100)
    torch.cuda.manual_seed(30100)
    torch.cuda.manual_seed_all(30100)
    
    # Generate negative sample pairs
    test_neg_src = test_pos_src
    test_neg_dst = torch.randint(0, g.num_nodes(ntype='user'), (g.num_edges(etype='friend'),))
    test_src = torch.cat([test_pos_src, test_neg_src])
    test_dst = torch.cat([test_pos_dst, test_neg_dst])
    test_labels = torch.cat(
        [torch.ones_like(test_pos_src), torch.zeros_like(test_neg_src)])
    test_preds = []
    
    for i in range(len(test_src)):                                                                          # Calculate cosine similarity
        test_preds.append((F.cosine_similarity(user_emb[test_src[i]], user_emb[test_dst[i]], dim=0)))

    # Calculate AUC and Average Precision
    auc = sklearn.metrics.roc_auc_score(test_labels.detach().numpy(), torch.tensor(test_preds))
    ap = sklearn.metrics.average_precision_score(test_labels.detach().numpy(), torch.tensor(test_preds))
    print('Link Prediction AUC:', auc)
    print("average_precision AP:", ap)

    # Calculate Top-k accuracy
    user_emb_norm = torch.norm(user_emb, dim=-1, keepdim=True)
    dot_numerator = torch.mm(user_emb, user_emb.t())
    dot_denominator = torch.mm(user_emb_norm, user_emb_norm.t())
    sim = (dot_numerator / dot_denominator )

    # Initialize similarity matrix 
    user_number=g.num_nodes(ntype='user')                                                 # Number of user nodes  
    cos=[[-1]*user_number for i in range(user_number) ]                                   # Initialize similarity matrix
    for i in range(g.num_nodes(ntype='user')):
        sim[i][i]=-1                                                                      # Set self-connections to -1
        if i in friend_ture.keys():                                                       # If user is in friend dictionary
            x=friend_ture[i]                                                              # Iterate over friends
            for j in x:                                                        
                sim[i][j]=-1                                                              # Set friendship relationships to -1



    friend_test_true={}                                             # Process test positive samples
    test_pos_src=list(test_pos_src.numpy())                         # Convert to list
    test_pos_dst=list(test_pos_dst.numpy())
    for i in range(len(test_pos_src)):                              # Update friend dictionary
        if test_pos_src[i] in friend_test_true.keys():
            friend_test_true[test_pos_src[i]]=friend_test_true[test_pos_src[i]]+[test_pos_dst[i]]
        else:
            friend_test_true[test_pos_src[i]]=[test_pos_dst[i]]

    for i in range(len(test_pos_dst)):
        if test_pos_dst[i] in friend_test_true.keys():
            friend_test_true[test_pos_dst[i]]=friend_test_true[test_pos_dst[i]]+[test_pos_src[i]]
        else:
            friend_test_true[test_pos_dst[i]]=[test_pos_src[i]]


    y_true=[]
    y_score=[]
    for i in friend_test_true.keys():
        y_true.append( friend_test_true[i])
        y_score.append(sim[i])

    
    k=[1,5,10,15,20] #top-k                                    # start counting top-k

    right_k=[0 for i in range(len(k))]                         # Initialize correct count
    for i in range(len(y_true)):
        sim_i = y_score[i].cpu().detach().numpy()              # Get similarity
        for j in range(len(k)):
            s = sim_i.argsort()[-k[j]:][::-1]                  # Get Top-k similarity indices
            if set(list(s)) & set(y_true[i]):
                right_k[j]+=1                                  # Increase correct count
    top_k=[0 for i in range(len(k))]

    # Print Top-k accuracy
    for j in range(len(k)):
        top_k[j]=right_k[j]/len(y_true)                                   # Calculate accuracy
        print("Top ",k[j],'accuracy score is:', right_k[j]/len(y_true))
    
        # Calculate F1 scores
        test_preds = torch.tensor(test_preds)                            # Convert a list into a tensor
        test_preds_binary = (test_preds > 0.5).float()                   # Binary prediction values

        # Positive F1 score 
        f1_pos = sklearn.metrics.f1_score(test_labels.detach().numpy(), test_preds_binary.detach().numpy(), pos_label=1)

        # Negative F1 score 
        f1_neg = sklearn.metrics.f1_score(test_labels.detach().numpy(), test_preds_binary.detach().numpy(), pos_label=0)

        # Macro average F1 score
        f1_macro = sklearn.metrics.f1_score(test_labels.detach().numpy(), test_preds_binary.detach().numpy(), average='macro')


        #  Print results
        print('Link Prediction AUC:', auc)
        print("Average Precision (AP):", ap)
        print("F1 Score (Positive Class):", f1_pos)
        print("F1 Score (Negative Class):", f1_neg)
        print("F1 Score (Macro Average):", f1_macro)

        # Top-k The accuracy rate is partially omitted and remains unchanged. ...


    # k=10
    # right=0
    # for i in range(len(y_true)):
    #     sim_i=y_score[i].cpu().detach().numpy()
    #     s=sim_i.argsort()[-k:][::-1]
    #     if set(list(s))& set(y_true[i]):
    #         right+=1
    # print("Top ",k,'accuracy score is:', right/len(y_true))
    #
    #
    # k=1
    # right=0
    # for i in range(len(y_true)):
    #     sim_i=y_score[i].cpu().detach().numpy()
    #     s=sim_i.argsort()[-k:][::-1]
    #     if set(list(s))& set(y_true[i]):
    #         right+=1
    # print("Top ",k,'accuracy score is:', right/len(y_true))
    #
    # k=5
    # right=0
    # for i in range(len(y_true)):
    #     sim_i=y_score[i].cpu().detach().numpy()
    #     s=sim_i.argsort()[-k:][::-1]
    #     if set(list(s))& set(y_true[i]):
    #         right+=1
    # print("Top ",k,'accuracy score is:', right/len(y_true))
    #
    # k=15
    # right=0
    # for i in range(len(y_true)):
    #     sim_i=y_score[i].cpu().detach().numpy()
    #     s=sim_i.argsort()[-k:][::-1]
    #     if set(list(s))& set(y_true[i]):
    #         right+=1
    # print("Top ",k,'accuracy score is:', right/len(y_true))
    #
    # k=20
    # right=0
    # for i in range(len(y_true)):
    #     sim_i=y_score[i].cpu().detach().numpy()
    #     s=sim_i.argsort()[-k:][::-1]
    #     if set(list(s))& set(y_true[i]):
    #         right+=1
    # print("Top ",k,'accuracy score is:', right/len(y_true))

    # return auc, ap,top_k
    return auc, ap,top_k, f1_pos, f1_neg, f1_macro

