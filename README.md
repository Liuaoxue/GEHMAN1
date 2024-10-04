# GEHMAN
Before to execute GEHMAN necessary to install the following packages:
<br/>
``pip install dgl``
<br/>
``pip install torch``
<br/>

## Requirements

- numpy ==1.26.4
- torch ==2.1.2
- dgl ==2.4.0

### Basic Usage

- --run  train.py to train the GEHMAN. and it probably need at least 11G GPU memory 
- --run  test.py to estimate the performance of GEHMAN based on the user representations that we learned during our experiments. You can also use this code to individually test the effects of your own learned representation.

### Miscellaneous

*Note:* This is only a reference implementation of GEHMAN. Our code implementation is partially based on the DGL library, for which we are grateful.

### Code directory

#### data

​	The index relationship, edge features and node features of the six cities after preprocessing are stored under the data folder.

#### config.py     

​	A command-line parameter parser for training GEHMAN model is defined, including parameters such as cuda specifying GPU, city selecting data set city, epochs setting training rounds, lr setting learning rate, multihead specifying the number of heads of multi-head attention mechanism, and three regularization parameters lambda_1, lambda_2 and lambda_3.

#### dataset.py   

​	The construction of heterogeneous multiple graphs and the initialization of points and edges are realized.

#### train.py       

​	It is used to train and verify the model, save the optimal model, and output the verification results AUC, AP, F1 score, etc.

#### model.py    

​	It is used to train and verify the model, save the optimal model, and output the verification results AUC, AP, F1 scores, etc. This file implements a link prediction model for heterogeneous graphs, in which the Edge-level module is used to deal with the neural network module of edge features in graphs. It calculates attention weight according to the characteristics of source node, target node and edge, aggregates the characteristics of nodes through message passing mechanism, and finally outputs the characteristic representation of each edge. Semantic-level module is used to aggregate features in multiple headers. The input features are transformed by linear layer and activation function, and the weight of each header is calculated by softmax function, so that different headers are weighted and summed to generate aggregated node features. The Heterogeneous Graph model uses Edge-level and Semantic-level modules to deal with different types of nodes and edges in the graph, and finally generates the representation of nodes through multi-level message transmission.

#### test.py      

​	Use the optimal model to test on the test set.

#### utils.py    

​	Some tool classes are defined to realize the training and evaluation of link prediction on heterogeneous graphs. The main functions include generating positive and negative sample edges, constructing negative sample graph, calculating the similarity between nodes by cosine similarity, using contrast loss and edge loss for model training, and evaluating the model performance by AUC, average accuracy (AP), Top-k accuracy and F1 score.
