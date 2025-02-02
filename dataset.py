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


np.random.seed(30100)
random.seed(30100)


# Set all random seeds for reproducibility.
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# Load and preprocess data for the specified city and node feature dimension
def data(d_node,city):

    #Load edge and node attribute data from .npy files
    friend_list_index = np.load('data/'+city+'/friend_list_index.npy')
    friend_edge_attr = np.load('data/'+city+'/friend_edge_attr.npy')
    visit_list_edge_tensor = np.load('data/'+city+'/visit_list_edge_tensor.npy')
    visit_list_edge_attr = np.load('data/'+city+'/visit_list_edge_attr.npy')
    revisit_list_edge_tensor = np.load('data/'+city+"/revisit_list_edge_tensor.npy")
    revisit_list_edge_attr = np.load('data/'+city+"/revisit_list_edge_attr.npy")
    live_with_edge = np.load('data/'+city+"/live_with_edge.npy")
    live_with_attr = np.load('data/'+city+"/live_with_attr.npy")
    re_live_with_edge=np.array([live_with_edge[1],live_with_edge[0]])
    re_live_with_attr=live_with_attr
    # re_live_with_edge = np.load('data/'+city+"/re_live_with_edge.npy")
    # re_live_with_attr = np.load('data/'+city+"/re_live_with_attr.npy")
    co_occurrence_list_index = np.load('data/'+city+'/co_occurrence_list_index.npy')
    co_occurrence_attr = np.load('data/'+city+'/co_occurrence_attr.npy')
    class_same_edge_tensor = np.load('data/'+city+"/class_same_edge.npy")
    class_same_edge_attr = np.load('data/'+city+"/class_same_edge_attr.npy")
    Active_association_edge_attr = np.load('data/'+city+'/Active association edge_attr.npy')
    Active_association_edge_tensor = np.load('data/'+city +'/Active association edge_tensor.npy')
    Co_visiting_edge_attr = np.load('data/'+city+'/Co-visiting edge_attr.npy')
    Co_visiting_edge_list_index = np.load('data/'+city +'/Co-visiting edge_list_index.npy')


    # Prepare training, validation, and test sets
    all_index = list(np.arange(0, friend_list_index.shape[1], 1))
    train_rate = 0.8
    train_len = round(friend_list_index.shape[1] * train_rate)
    train_edge_index = sorted(random.sample(all_index, train_len))
    test_edge_index = sorted(list(set(all_index).difference(set(train_edge_index))))
    

    # Select a verification set from the test set
    val_rate = 0.5                                   # Select 50% as the verification set
    val_len = round(len(test_edge_index) * val_rate)
    val_edge_index = sorted(random.sample(test_edge_index, val_len))
    test_edge_index = sorted(list(set(test_edge_index).difference(set(val_edge_index))))


    friend_list_index_val = []
    friend_edge_attr_val = []
    friend_list_index_train = []
    friend_edge_attr_train = []
    friend_list_index_test = []
    friend_edge_attr_test = []

    # Populate training edges
    for i in train_edge_index:
        friend_list_index_train.append((friend_list_index[0][i], friend_list_index[1][i]))
        friend_list_index_train.append((friend_list_index[1][i], friend_list_index[0][i]))
        friend_edge_attr_train.append(list(friend_edge_attr[i]))
        friend_edge_attr_train.append(list(friend_edge_attr[i]))

    # Populate validation edges
    for i in val_edge_index:
        friend_list_index_val.append((friend_list_index[0][i], friend_list_index[1][i]))
        friend_edge_attr_val.append(list(friend_edge_attr[i]))
        
    # Populate test edges
    for i in test_edge_index:
        friend_list_index_test.append((friend_list_index[0][i], friend_list_index[1][i]))
        friend_edge_attr_test.append(list(friend_edge_attr[i]))
        

    # Convert lists to tensors
    friend_list_index_train = torch.tensor(np.array(friend_list_index_train)).t().contiguous()
    friend_list_index_test = torch.tensor(np.array(friend_list_index_test)).t().contiguous()
    friend_edge_attr_train = np.array(friend_edge_attr_train)
    friend_edge_attr_test = np.array(friend_edge_attr_test)
    friend_list_index_val = torch.tensor(np.array(friend_list_index_val)).t().contiguous()
    friend_edge_attr_val = np.array(friend_edge_attr_val)


    # Create graph data structure with node types and edge types
    graph_data = {
        ('user', 'friend', 'user'): (torch.tensor(friend_list_index_train[0]), torch.tensor(friend_list_index_train[1])),
        ('user', 'visit', 'poi'): (torch.tensor(visit_list_edge_tensor[0]), torch.tensor(visit_list_edge_tensor[1])),
        ('poi', 'co_occurrence', 'poi'): (torch.tensor(co_occurrence_list_index[0]), torch.tensor(co_occurrence_list_index[1])),
        ('poi', 'live_with', 'user'): (torch.tensor(live_with_edge[0]), torch.tensor(live_with_edge[1])),
        ('user', 're_live_with', 'poi'): (torch.tensor(re_live_with_edge[0]), torch.tensor(re_live_with_edge[1])),
        ('poi', 're_visit', 'user'): (torch.tensor(revisit_list_edge_tensor[0]), torch.tensor(revisit_list_edge_tensor[1])),
        ('poi', 'class_same', 'poi'): (torch.tensor(class_same_edge_tensor[0]), torch.tensor(class_same_edge_tensor[1])),
        ('user', 'Active_association', 'poi'): (torch.tensor(Active_association_edge_tensor[0]), torch.tensor(Active_association_edge_tensor[1])),
        ('user', 'Co_visiting', 'user'): (torch.tensor(Co_visiting_edge_list_index[0]), torch.tensor(Co_visiting_edge_list_index[1]))
    }
    g = dgl.heterograph(graph_data)
    print(g.ntypes)
    print("user_num_nodes", g.num_nodes('user'))
    print("poi_num_nodes", g.num_nodes('poi'))
    
    # Random initialization of node features
    g.nodes['user'].data['u_fe'] = torch.tensor(np.random.randint(0, 10, (g.num_nodes('user'), d_node)), dtype=torch.float32)
    g.nodes['poi'].data['p_fe'] = torch.tensor(np.random.randint(0, 10, (g.num_nodes('poi'), d_node)), dtype=torch.float32)
    g.edges['friend'].data['f_fe'] = torch.tensor(friend_edge_attr_train, dtype=torch.float32)
    g.edges['visit'].data['v_fe'] = torch.tensor(visit_list_edge_attr, dtype=torch.float32)
    g.edges['co_occurrence'].data['c_fe'] = torch.tensor(co_occurrence_attr, dtype=torch.float32)
    g.edges['live_with'].data['l_fe'] = torch.tensor(live_with_attr, dtype=torch.float32)
    g.edges['re_live_with'].data['rl_fe'] = torch.tensor(re_live_with_attr, dtype=torch.float32)
    g.edges['class_same'].data['cl_fe'] = torch.tensor(class_same_edge_attr, dtype=torch.float32)
    g.edges['re_visit'].data['r_fe'] = torch.tensor(revisit_list_edge_attr, dtype=torch.float32)
    g.edges['Active_association'].data['Ac_fe'] = torch.tensor(Active_association_edge_attr, dtype=torch.float32)
    g.edges['Co_visiting'].data['Co_fe'] = torch.tensor(Co_visiting_edge_attr, dtype=torch.float32)

    return g, friend_list_index_test,friend_list_index_val





