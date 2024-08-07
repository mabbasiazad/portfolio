+++
title = 'Compatible Activation and Loss Functions'
date = 2023-11-01
author= ["Mehdi Azad"]
summary = "Choosing loss function and last layer activation function for neural network "
+++

Computing the loss for classification tasks involves the following steps: 

1. Using softmax to compute probabilities from logits/scores.

2. Taking the logarithm of the probabilities.

3. Computing the cross-entropy between predictions and labels.

It's more efficient to merge steps 1 and 2, performing them simultaneously by computing log-softmax. 

In PyTorch, these steps might be performed partly within activation functions and partly within loss functions. 
This variation in implementation necessitates choosing compatible activation and loss functions.  

## multi-class classificatoin

no activation function => `torch.nn.CrossEntropyLoss`

`torch.nn.LogSoftmax`  => `torch.nn.NLLLoss`

`torch.nn.Softmax` => `torch.log` => `torch.nn.NLLLoss`      # not efficient

-------------------------------------------------

## binary classification

no activation function => `torch.nn.BCEWithLogitsLoss`

`torch.nn.Sigmoid`  => `torch.nn.BCELoss`   # not efficient

