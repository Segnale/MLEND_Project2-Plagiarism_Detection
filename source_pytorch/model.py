# torch imports
import torch.nn.functional as F
import torch.nn as nn


## Done: Complete this classifier
class BinaryClassifier(nn.Module):
    """
    Define a neural network that performs binary classification.
    The network should accept your number of features as input, and produce 
    a single sigmoid value, that can be rounded to a label: 0 or 1, as output.
    
    Notes on training:
    To train a binary classifier in PyTorch, use BCELoss.
    BCELoss is binary cross entropy loss, documentation: https://pytorch.org/docs/stable/nn.html#torch.nn.
    """
    
    ## Done: Define the init function, the input params are required (for loading code in train.py to work)
    def __init__(self, input_features, hidden_dim, output_dim):
        """
        Initialize the model by setting up linear layers.
        Use the input parameters to help define the layers of your model.
        :param input_features: the number of input features in your training/test data
        :param hidden_dim: helps define the number of nodes in the hidden layer(s)
        :param output_dim: the number of outputs you want to produce
        """
        super(BinaryClassifier, self).__init__()

        # define any initial layers, here
        self.layer1 = nn.Linear(in_features=input_features, out_features =hidden_dim)
        self.layer2 = nn.Linear(in_features=hidden_dim, out_features =output_dim)
        self.layer3 = nn.Linear(in_features=output_dim, out_features=1)
        self.sig = nn.Sigmoid()

    
    ## Done: Define the feedforward behavior of the network
    def forward(self, x):
        """
        Perform a forward pass of our model on input features, x.
        :param x: A batch of input features of size (batch_size, input_features)
        :return: A single, sigmoid-activated value as output
        """
        
        # define the feedforward behavior
        batch_size = x[0,:]
        input_features = x[1:,:]
        
        l1 = self.layer1(input_features)
        l2 = self.layer2(l1)
        out = self.dense(l2)
        
        return self.sig(out.squeeze())
    