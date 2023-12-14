+++
title = 'Activation Function of Last Layer ⇒ Loss Fuction'
date = 2023-12-13
author= ["Mehdi Azad"]
summary = "Choosing loss function and last layer activation function for neural network "
+++



## multi-class classificatoin

no activation function —> nn.CrossEntropyLoss 

F.log_softmax  —> nn.NLLLoss

softmax —> log —> NLLLoss      # not efficient

-------------------------------------------------

## binary classification

no activation function —> nn.BCEWithLogitsLoss

sigmoid  —> BCELoss   # not efficient

