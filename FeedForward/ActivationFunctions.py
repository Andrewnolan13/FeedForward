
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))
#Exp Relu
def ExpRelu(x):
    return (x<0)*(np.exp(x) - 1) + (x>=0)*x
    
def d_ExpRelu(x):
    return (ExpRelu(x)<0)*(ExpRelu(x)+1)+(ExpRelu(x)>=0)*1
#Tanh
def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    
def d_tanh(x):
    return 1-tanh(x)*tanh(x)
#Softplus
def softplus(x):
    return np.log(1+np.exp(x))
def d_sofplus(f):
    return 1-np.exp(-softplus(f))

def relu(x):
    return (x>0)*x + 0
def d_relu(x):
    return (x>0)*1+0

def linear(x):
    return x
def d_linear(x):
    return np.ones_like(x)