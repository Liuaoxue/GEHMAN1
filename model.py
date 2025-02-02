import dgl
import numpy as np
import torch
from torch import nn
import dgl.nn as dglnn
import torch.nn as nn
import dgl.function as fn
import random
import tqdm
import sklearn.metrics
from torch import cosine_similarity
import torch.nn.functional as F
from dgl.nn.functional import edge_softmax
from dgl.utils import expand_as_pair
from dgl.nn.pytorch.utils import Identity
import dgl.function as fn
import torch as th
from torch import nn
from utils import *
from torch.nn import init

# Edge-level model
class Edge_level(nn.Module):
    """Edge-level neural network module for processing edge features in a heterogeneous graph.
    Args:
        in_feats (int or tuple): Input feature size for source and destination nodes.
        out_feats (int): Output feature size for edges.
        num_heads (int): Number of attention heads.
        edge_dim (int, optional): Dimensionality of edge features. Default is 5.
        feat_drop (float, optional): Dropout rate for node features. Default is 0.
        attn_drop (float, optional): Dropout rate for attention weights. Default is 0.
        negative_slope (float, optional): Negative slope for Leaky ReLU. Default is 0.2.
        residual (bool, optional): Use residual connections if True. Default is False.
        activation (callable, optional): Activation function to apply. Default is None.
        allow_zero_in_degree (bool, optional): Allow zero in-degree nodes. Default is False.
        bias (bool, optional): Include bias in linear layers if True. Default is True.
    """
    
    def __init__(self,  in_feats,  out_feats,  num_heads, edge_dim=5,  feat_drop=0., attn_drop=0., negative_slope=0.2,residual=False, activation=None,allow_zero_in_degree=False, bias=True):
        super(Edge_level, self).__init__()
        # Initialize parameters
        
        self.edge_dim = edge_dim
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        # Define linear transformations
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.fc_edg = nn.Linear(self.edge_dim, self.edge_dim * num_heads)
        
        # Define attention parameters
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_edg = nn.Parameter(th.FloatTensor(size=(1, num_heads, self.edge_dim)))
        
        # Define output linear layer
        self.lin_out = nn.Linear(self.edge_dim + self._out_feats, self._out_feats)
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(th.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters for the model."""
        gain = nn.init.calculate_gain('relu')
        
        # Initialize weights for layers
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
            nn.init.xavier_normal_(self.lin_out.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_edg, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
         """Set whether to allow nodes with zero in-degree.
            Args:
                set_value (bool): Boolean value indicating whether to allow zero in-degree.
        """
        
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, edge_fea,):
        """Forward pass for the edge-level model.
        Args:
            graph (dgl.DGLGraph): DGL graph containing node and edge data.
            feat (tuple): Tuple of source and destination node features.
            edge_fea (torch.Tensor): Edge feature tensor.
        Returns:
            torch.Tensor: Output features for the edges.
        """
        
        with graph.local_scope():   # Process node and edge features
            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]  #
                edge_fea_shape = edge_fea.shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                edge_fea = self.feat_drop(edge_fea)
                if not hasattr(self, 'fc_src'):
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
                    edge_fea = self.fc_edg(edge_fea).view(
                        *edge_fea_shape, self._num_heads, self.edge_dim)
                else:
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
                    dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]
                    
            # Compute edge weights
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            e_e = (edge_fea * self.attn_edg).sum(dim=-1).unsqueeze(-1)
            graph.edata['_edge_weight'] = edge_fea
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            data_e = graph.edata.pop('e')
            data_e = data_e + e_e
            e = self.leaky_relu(data_e)
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph,e))
            
            # message passing
            def edge_udf(edges):
                return {'he': torch.mul(edges.data['a'], edges.data['_edge_weight'])}

            graph.update_all(edge_udf, fn.sum('he', 'ft_f'))
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            f = graph.dstdata['ft_f']
            rst = graph.dstdata['ft']
            rst = self.lin_out(torch.cat([f, rst], dim=-1))
            
            # residual
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(
                    *dst_prefix_shape, -1, self._out_feats)
                rst = rst + resval
                
            # bias
            if self.bias is not None:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)
            # activation
            if self.activation:
                rst = self.activation(rst)
            return rst

# Semantic-level model
class Semantic_level(nn.Module):

    """Semantic-level neural network module for aggregating features across multiple heads.
    Args:
        in_size (int): Input feature size for each head.
        num_head (int): Number of attention heads.
        hidden_size (int, optional): Hidden layer size in the semantic model. Default is 128.
    """
    def __init__(self, in_size, num_head, hidden_size=128):
        super(Semantic_level, self).__init__()
        self.Linear1 = nn.Linear(in_size * num_head, hidden_size)
        self.tanh = nn.Tanh()
        self.Linear2 = nn.Linear(hidden_size, 1, bias=False)
        self.num_head = num_head
        self.in_size = in_size
    def forward(self, z):

         """Forward pass for the semantic-level model.

        Args:
            z (torch.Tensor): Input feature tensor from multiple heads.

        Returns:
            torch.Tensor: Aggregated features for each head.
        """
        # Process input and return aggregated features
        
        z = th.stack(z, dim=0)
        z = z.transpose(1, 0, )
        z = th.reshape(z, (z.shape[0], z.shape[1], z.shape[2] * z.shape[3]))
        w = self.Linear1(z)
        w = self.tanh(w)
        w = self.Linear2(w).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)
        beta = (beta * z).sum(1)
        beta = th.reshape(beta, (beta.shape[0], self.num_head, self.in_size))
        return beta

# Heterogeneous graph model
class HeteroGraph(nn.Module):

    """Heterogeneous graph model for processing different types of nodes and edges.

    Args:
        mods (dict): Dictionary of edge-level models for each relationship type.
        in_size_sem (int): Input feature size for semantic attention.
        num_head (int): Number of attention heads.
    """
    def __init__(self, mods, in_size_sem, num_head):
        super(HeteroGraph, self).__init__()
        self.semantic_attention1 = Semantic_level(in_size=in_size_sem, num_head=num_head)
        self.semantic_attention2 = Semantic_level(in_size=in_size_sem, num_head=num_head)
        self.mods = nn.ModuleDict(mods)
        for _, v in self.mods.items():
            set_allow_zero_in_degree_fn = getattr(v, 'set_allow_zero_in_degree', None)
            if callable(set_allow_zero_in_degree_fn):
                set_allow_zero_in_degree_fn(True)

    def forward(self, g, inputs, edge_attr, mod_args=None, mod_kwargs=None):

        """Forward pass for the heterogeneous graph model.

        Args:
            g (dgl.DGLGraph): DGL graph containing node and edge data.
            inputs (dict): Node features for source and destination nodes.
            edge_attr (dict): Edge attributes for each relationship type.
            mod_args (dict, optional): Additional arguments for edge-level models. Default is None.
            mod_kwargs (dict, optional): Additional keyword arguments for edge-level models. Default is None.

        Returns:
            dict: Aggregated output features for target nodes.
        """
        if mod_args is None:
            mod_args = {}
        if mod_kwargs is None:
            mod_kwargs = {}
        outputs = {nty: [] for nty in g.dsttypes}
        if isinstance(inputs, tuple) or g.is_block:
            if isinstance(inputs, tuple):
                src_inputs, dst_inputs = inputs
            else:
                src_inputs = inputs
                dst_inputs = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}

            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if stype not in src_inputs or dtype not in dst_inputs:
                    continue
                dstdata = self.mods[etype](
                    rel_graph,
                    (src_inputs[stype], dst_inputs[dtype]),
                    edge_attr[etype],
                    *mod_args.get(etype, ()),
                    **mod_kwargs.get(etype, {}))
                outputs[dtype].append(dstdata)
        else:
            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if stype not in inputs:
                    continue
                dstdata = self.mods[etype](
                    rel_graph,
                    (inputs[stype], inputs[dtype]),
                    edge_attr[etype],
                    *mod_args.get(etype, ()),
                    **mod_kwargs.get(etype, {}))
                outputs[dtype].append(dstdata)
        rsts = {}
        for nty, alist in outputs.items():
            if len(alist) != 0:
                if nty == 'user':
                    rsts[nty] = self.semantic_attention1(alist)
                else:
                    rsts[nty] = self.semantic_attention2(alist)
        return rsts

# Heterogeneous Message Passing Neural Network
class HMGNN(nn.Module):

    """Heterogeneous Message Passing Neural Network for learning node representations.

    Args:
        in_feats (int): Input feature size for nodes.
        hid_feats (int): Hidden feature size for nodes.
        out_feats (int): Output feature size for nodes.
        rel_names (list): List of relationship types.
        num_heads (int): Number of attention heads.
    """
    def __init__(self, in_feats, hid_feats, out_feats, rel_names, num_heads):
        super().__init__()
        self.conv1 = HeteroGraph({
            rel: Edge_level(in_feats, hid_feats, num_heads=num_heads, )
            for rel in rel_names}, in_size_sem=hid_feats, num_head=num_heads)
        self.conv2 = HeteroGraph({
            rel: Edge_level(hid_feats * num_heads, out_feats, num_heads=num_heads, )
            for rel in rel_names}, in_size_sem=out_feats, num_head=num_heads)
        self.lin = nn.Linear(out_feats * num_heads, out_feats)
        self.lin2 = nn.Linear(out_feats, out_feats)
        self.relu = nn.ReLU()

    def forward(self, graph, inputs, edge_attr):

         """Forward pass for the HMGNN.
        Args:
            graph (dgl.DGLGraph): DGL graph containing node and edge data.
            inputs (dict): Node features for processing.
            edge_attr (dict): Edge attributes for each relationship type.

        Returns:
            dict: Node representations after processing.
        """
        h = self.conv1(graph, inputs, edge_attr)
        h = {k: v.reshape(v.shape[0], -1) for k, v in h.items()}
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h, edge_attr)
        h = {k: v.reshape(v.shape[0], -1) for k, v in h.items()}
        h = {k: F.relu(v) for k, v in h.items()}
        h = {k: self.lin(v) for k, v in h.items()}
        h = {k: self.lin2(v) for k, v in h.items()}
        return h

# Main model
class Model(nn.Module):

    """Main model integrating the heterogeneous graph neural network and prediction.

    Args:
        in_features (int): Input feature size for nodes.
        hidden_features (int): Hidden feature size for nodes.
        out_features (int): Output feature size for nodes.
        rel_names (list): List of relationship types.
        num_heads (int): Number of attention heads.
    """
    def __init__(self, in_features, hidden_features, out_features, rel_names, num_heads):
        super().__init__()
        self.rel_names = rel_names
        self.sage = HMGNN(in_features, hidden_features, out_features, rel_names, num_heads)
        self.pred = HeteroDotProductPredictor()
        self.lin = nn.Linear(out_features, out_features)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.fc_list_node = nn.ModuleList([nn.Linear(feats_dim, in_features, bias=True)
                                           for feats_dim in [128, 128]])
        self.fc_list_edg = nn.ModuleList([nn.Linear(feats_dim, 5, bias=True) for feats_dim in   [5, 4, 2, 1, 1, 2, 4,4,5]])
        for fc in self.fc_list_node:
            nn.init.xavier_normal_(fc.weight,gain=1.414)
        for fc in self.fc_list_edg:
            nn.init.xavier_normal_(fc.weight,gain=1.414)

    def predict(self, h, pos_edge_index, neg_edge_index):
    """Predict the likelihood of edges between node pairs.

        Args:
            h (torch.Tensor): Node representations.
            pos_edge_index (torch.Tensor): Positive edge indices.
            neg_edge_index (torch.Tensor): Negative edge indices.

        Returns:
            torch.Tensor: Predicted scores for edges.
        """
        
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = cosine_similarity(h[edge_index[0]], h[edge_index[1]])
        logits_2 = self.relu(logits)
        return logits_2

    def forward(self, g, neg_g, node_feat, edge_attr, etype):
        """Forward pass for the main model.

        Args:
            g (dgl.DGLGraph): DGL graph containing node and edge data.
            neg_g (dgl.DGLGraph): Negative graph for contrastive learning.
            node_feat (dict): Node features for input.
            edge_attr (dict): Edge attributes for each relationship type.
            etype (str): Relationship type for edge predictions.

        Returns:
            tuple: Predictions for positive and negative edges, along with node representations.
        """
        
        feat2 = {}
        feat2['user'] = self.relu(self.fc_list_node[0](node_feat['user']))
        feat2['poi'] = self.relu(self.fc_list_node[1](node_feat['poi']))
        i = 0
        edge_attr_new = {}
        for edg in self.rel_names:
            edge_attr_new[edg] = self.relu(self.fc_list_edg[i](edge_attr[edg]))
            i += 1
        h = self.sage(g, feat2, edge_attr_new)
        return self.pred(g, h, etype), self.pred(neg_g, h, etype), h, contrastive_loss(h['user'], g)

