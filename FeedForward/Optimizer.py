import numpy as np
import random

def _shuffle_and_partition_array(arr1:np.ndarray,arr2:np.ndarray, k:int)->np.ndarray:
    # Get the number of rows in the array
    n = arr1.shape[0]
    indices = list(range(n))
    random.shuffle(indices)
    partition_size = n // k
    partitions1 = np.array_split(arr1[indices], k)
    partitions2 = np.array_split(arr2[indices], k)
    # Check if the last partition has a small size and omit for homogeneity
    if len(partitions1[-1]) != partition_size:
        partitions1.pop()
        partitions2.pop()

    return partitions1,partitions2

class Adam:
    def __init__(self,NeuralNetwork,beta1=0.9, beta2=0.999, epsilon=1e-8,method='BGD'):
        # For the dumbasses (me)
        if isinstance(method,dict):
            if len(method.keys())>1:
                raise ValueError('Only acceptable dict input is {"Mini":batch_size}. batch_size must be a Natural Number.')
            try:
                if method['Mini']<=0 or method['Mini']>=len(NeuralNetwork.X):
                    raise ValueError(f'batch_size must be a Natural Number smaller than {len(NeuralNetwork.X)}.')
            except:
                raise ValueError('Only acceptable dict input is {"Mini":batch_size}. batch_size must be a Natural Number.')
            method['Mini'] = int(method['Mini'])
            print(f"Excluding {len(NeuralNetwork.X) - method['Mini']*(len(NeuralNetwork.X)//method['Mini'])} Training examples per Epoch. Redefine batch_size if neccessary.")
        elif method not in ['BGD','SGD']:
            raise ValueError('Please choose one method from the following:\n"BGD".\n"SGD".\n{"Mini":batch_size} splits data into batches, updates parameters num_batches times per epoch.')

        self.method = method
        self.NeuralNetwork = NeuralNetwork
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        # Initialize first and second moments for weights and biases
        self.m_W = {k: np.zeros_like(self.NeuralNetwork.W[k]) for k in self.NeuralNetwork.W}
        self.v_W = {k: np.zeros_like(self.NeuralNetwork.W[k]) for k in self.NeuralNetwork.W}
        self.m_B = {k: np.zeros_like(self.NeuralNetwork.B[k]) for k in self.NeuralNetwork.B}
        self.v_B = {k: np.zeros_like(self.NeuralNetwork.B[k]) for k in self.NeuralNetwork.B}
    
    def _update(self,X_batch:np.ndarray, y_batch:np.ndarray, learning_rate:float):
        '''only intended for use within the step method of the Adam class.'''
        self.NeuralNetwork.backward(X_batch, y_batch, learning_rate)
        m = len(self.NeuralNetwork.X)//len(X_batch)
        # Update moments for weights and biases
        for k in self.NeuralNetwork.W:
            self.m_W[k] = self.beta1 * self.m_W[k] + (1 - self.beta1) * self.NeuralNetwork.dL_dW[k] / m
            self.v_W[k] = self.beta2 * self.v_W[k] + (1 - self.beta2) * (self.NeuralNetwork.dL_dW[k] / m) ** 2
            m_hat_W = self.m_W[k] / (1 - self.beta1 ** (self.NeuralNetwork.epoch + 1))
            v_hat_W = self.v_W[k] / (1 - self.beta2 ** (self.NeuralNetwork.epoch + 1))
            self.NeuralNetwork.W[k] -= learning_rate * m_hat_W / (np.sqrt(v_hat_W) + self.epsilon)
            
            self.m_B[k] = self.beta1 * self.m_B[k] + (1 - self.beta1) * self.NeuralNetwork.dL_dB[k] / m
            self.v_B[k] = self.beta2 * self.v_B[k] + (1 - self.beta2) * (self.NeuralNetwork.dL_dB[k] / m) ** 2
            m_hat_B = self.m_B[k] / (1 - self.beta1 ** (self.NeuralNetwork.epoch + 1))
            v_hat_B = self.v_B[k] / (1 - self.beta2 ** (self.NeuralNetwork.epoch + 1))
            self.NeuralNetwork.B[k] -= learning_rate * m_hat_B / (np.sqrt(v_hat_B) + self.epsilon)
    
    def step(self,learning_rate:float):
        '''only goes one step in the direction of the chosen method. Thus learning_rate is a scalar.'''
        if self.method == 'BGD':
            self._update(self.NeuralNetwork.X, self.NeuralNetwork.Y, learning_rate)
        elif self.method == 'SGD':
            indices = list(range(len(self.NeuralNetwork.X)))
            random.shuffle(indices)
            for i in indices:
                self._update(self.NeuralNetwork.X[i:i+1], self.NeuralNetwork.Y[i:i+1], learning_rate)
        else:
            Xpartitioins,Ypartitions = _shuffle_and_partition_array(self.NeuralNetwork.X,self.NeuralNetwork.Y,len(self.NeuralNetwork.X)//self.method['Mini']+1)
            for X_batch, y_batch in zip(Xpartitioins,Ypartitions):
                self._update(X_batch, y_batch, learning_rate)

class BGD:
    def __init__(self,NeuralNetwork):
        self.NeuralNetwork = NeuralNetwork
    def step(self,learning_rate:float):
        self.NeuralNetwork.backward(self.NeuralNetwork.X,self.NeuralNetwork.Y,learning_rate)