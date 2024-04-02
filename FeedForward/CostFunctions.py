import numpy as np

"""
Reduce complexity by not doing the forwar pass inside the Loss function. Only do Forwards when absolutely necessary.

Can probaby remove the inputs from the loss functions - I don't think I will ever want to calculate the loss on a subset, will I? Leave open for now. Actually, I do want the inputs and outputs if I want to get the gradient
of a subset. What is the point in doing subsets if you are getting the gradient of the entirety of the set? Completely defeats the purpose.

Suppose I am not doing batches? Then it is a waste of resources to do the forward. For now, just do the forward and swallow the cost, but later can encorporate conditionality somehow. 
Maybe a good idea for a parameter of batch:bool = False ? need to think about that. 
"""
class MSE:
    def __init__(self):
        self.Loss = self.Loss
        self.dLoss = self.dLoss 

    def Loss(self,NeuralNetwork:object,Inputs:np.ndarray,Targets:np.ndarray,update:bool):
        #If the layers need to be updated or if the most recent calculation is the same - this rules out redundant calculations. 
        if update==True:
            Predictions = NeuralNetwork.forward(Inputs) 
        else:
            Predictions = NeuralNetwork.o[NeuralNetwork.number_hidden_layers+1]
        difference = Predictions-Targets
        Cost = difference*difference/(2*Inputs.shape[0])
        
        return Cost.sum()

    def dLoss(self,NeuralNetwork:object,Inputs:np.ndarray,Targets:np.ndarray,update:bool):
        #If the layers need to be updated or if the most recent calculation is the same - this rules out redundant calculations. 
        if update==True:
            Predictions = NeuralNetwork.forward(Inputs) 
        else:
            Predictions = NeuralNetwork.o[NeuralNetwork.number_hidden_layers+1]
        difference = Predictions-Targets
        grad = difference*NeuralNetwork.dg[NeuralNetwork.number_hidden_layers+1](NeuralNetwork.a[NeuralNetwork.number_hidden_layers+1])
        
        # return grad.mean(axis = 0) <-- take mean here or inside the backward pass?
        return grad

def CrossEntropy(NeuralNetwork:object,Inputs:np.ndarray,Targets:np.ndarray):
    Predictions = NeuralNetwork.forward(Inputs)
    Cost = -Targets*np.log(Predictions) - (1-Targets)*np.log(1-Predictions)
    return Cost.sum()

def L1(NeuralNetwork:object,Inputs:np.ndarray,Targets:np.ndarray):
    Predictions = NeuralNetwork.forward(Inputs)
    difference = Predictions-Targets
    Cost = np.abs(difference)
    return Cost.sum()

def L2(NeuralNetwork:object,Inputs:np.ndarray,Targets:np.ndarray):
    Predictions = NeuralNetwork.forward(Inputs)
    difference = Predictions-Targets
    Cost = difference*difference
    return Cost.sum()

def Huber(NeuralNetwork:object,Inputs:np.ndarray,Targets:np.ndarray,delta:float=1.0):
    Predictions = NeuralNetwork.forward(Inputs)
    difference = Predictions-Targets
    Cost = np.where(np.abs(difference) < delta, 0.5*difference*difference, delta*(np.abs(difference)-0.5*delta))
    return Cost.sum()
