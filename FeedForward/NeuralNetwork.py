#External modules
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

#Internal modules
from .Scaler import Scaler
from .CostFunctions import *
from .Optimizer import *
from .ActivationFunctions import *
class NeuralNetwork:
    def __init__(self, 
                 Inputs:np.ndarray, 
                 hidden_sizes:list[int], 
                 Targets:np.ndarray,
                 activation_functions:list[callable],
                 activation_function_derivs:list[callable],
                 CostFunction:object = None,
                 lbound:float=0.0, 
                 ubound:float= 1.0):
        '''
        Inputs:np.ndarray 
        hidden_sizes:list[int] 
        Targets:np.ndarray
        activation_functions:list[function]
        activation_function_derivs:list[function]
        CostFunction:function - Should take in self(the NN class), two np.ndarrays(scaled Inputs and Targets) and return a float. 

        *** for the Scaler class. Scales the inputs and targets to be between lbound and ubound ***
        lbound:float=0.0  
        ubound:float= 1.0
        '''
        self.Inputs = Inputs
        self.Targets = Targets
        self.input_size = Inputs.shape[1]
        self.hidden_sizes = hidden_sizes
        self.output_size = Targets.shape[1]
        self.number_hidden_layers = len(hidden_sizes)
        if CostFunction is None:
            self.CostFunction = MSE()
        else:
            self.CostFunction = CostFunction
        self.Loss = self.CostFunction.Loss
        self.dLoss = self.CostFunction.dLoss

        #Scale inputs and Targets
        self.Iscales = Scaler(Inputs,lbound=lbound,ubound = ubound)
        self.Tscales = Scaler(Targets,lbound=lbound,ubound = ubound)
        self.X = self.Iscales.scaledarray
        self.Y = self.Tscales.scaledarray

        #dicts make it easier to call the 1st layer layer_weight[1] instead of layer_weight[0]
        W = {}
        B = {}
        
        #first layers weights is first_layer_size x input_size Matrix. Biases is a first_layer_size vector
        W[1] = np.random.randn(self.hidden_sizes[0],self.input_size).astype('double')
        B[1] = np.random.randn(self.hidden_sizes[0],1).astype('double')
        
        #nth layers weights is n_size x n-1_size Matrix. Biases is a n_size vector
        i=0
        for size in hidden_sizes[1:]:
            W[i+2] = np.random.randn(size,self.hidden_sizes[i]).astype('double')
            B[i+2] = np.random.randn(size,1).astype('double')
            i+=1
            
        W[i+2] = np.random.randn(self.output_size,self.hidden_sizes[i]).astype('double')
        B[i+2] = np.random.randn(self.output_size,1).astype('double')
            
        self.W = W
        self.B = B
        
        #activation functions and their derivatives
        self.g = {}
        self.dg={}
        i=1

        for fn in activation_functions:
            self.g[i] = fn
            self.dg[i] = activation_function_derivs[i-1]
            i+=1

        #Reporting
        self.Epoch_Loss_Grad = np.array([[0,0,0]])
        self.epoch = 0
            
    def forward(self,x:np.ndarray):
        self.o = {}
        self.o[0] = x.copy()
        self.a = {}
        
        for k in range(self.number_hidden_layers+1):
            o_k = self.o[k].copy()
            self.a[k+1] = self.B[k+1] + self.W[k+1]@o_k
            self.o[k+1] = self.g[k+1](self.a[k+1])
            
        return self.o[self.number_hidden_layers+1]
    
    def backward(self,x:np.ndarray,y:np.ndarray,learning_rate:float):
        '''goes backwards by one step, therefore learning_rate is expecting a constant'''
        M = self.number_hidden_layers+1 #remove for sake of complexity after
        self.delta = {}
        # an array of deltas for each x in X. ie [dC/da for each a in a[M](x)] for each x in X. Refer to calculus
        self.delta[M] = self.dLoss(self = self.CostFunction,NeuralNetwork=self,Inputs = x,Targets = y,update = True)#.reshape(-1,self.input_size,1) <- Afraid to remove
        for k in range(self.number_hidden_layers,0,-1):
            dg_ak = self.dg[k](self.a[k])
            self.delta[k] = (self.W[k+1].T@self.delta[k+1])*(dg_ak)#.reshape(-1,dg_ak.shape[1],1) #<- Afraid to remove

        #Update parameters in each Layer
        ## these are stored to be passed to Optimizers
        self.dL_dW = {}
        self.dL_dB = {}
        for k in range(self.number_hidden_layers+1,0,-1):
            self.dL_dW[k] = (self.delta[k]*self.o[k-1].transpose([0,2,1])).mean(axis = 0) # self.dL_dW[k] = np.array([delta*o.T for delta,o in zip(self.delta[k],self.o[k-1])]).mean(axis = 0) <> took about 4 hours to figure out how to do this with numpy's own methods.
            self.dL_dB[k] = self.delta[k].mean(axis = 0)

            self.W[k] -= learning_rate*self.dL_dW[k]
            self.B[k] -= learning_rate*self.dL_dB[k]

            """
            The intention here is for each r_k by 1 vector in delta, matrix multiply it by the transpose of the r_{k-1} by 1 vector in o[k-1] to give a r_k by r_{k-1} matrix.
            r[k] is the number of neurons in the kth layer.
            remember that delta here is an array of r_k by 1 vectors, and o[k-1] is an array of r_{k-1} by 1 vectors. The array has the same size as the number of inputs in x. 
            Numpy broadcasting is really a pain in the arse to figure out.
            """
    
    def _backward(self,x:np.ndarray,y:np.ndarray,learning_rate:float): #<- want to keep this in incase I somehow got it right the first time. Will delete when cofident.
        '''goes backwards by one step, therefore learning_rate is expecting a constant'''
        M = self.number_hidden_layers+1 #remove for sake of complexity after
        delta = {}

        delta[M] = self.dLoss(self = self.CostFunction,NeuralNetwork=self,Inputs = x,Targets = y,update = True) # an array of deltas for each x in X
        for k in range(self.number_hidden_layers,0,-1):
            delta[k] = (self.W[k+1]@delta[k+1])*self.dg[k](self.a[k]) #an array of deltas for each x in X

        #Update parameters in each Layer
        ## these are stored to be passed to Optimizers
        self.dL_dW = {}
        self.dL_dB = {}
        for k in range(self.number_hidden_layers+1,0,-1):
            self.dL_dW[k] = (delta[k]@self.o[k-1].T).mean(axis = 0)
            self.dL_dB[k] = delta[k].mean(axis = 0)

            self.W[k] -= learning_rate*self.dL_dW[k]
            self.B[k] -= learning_rate*self.dL_dB[k]
        
    def Train(self,epochs:int,learning_rate:callable,Optimizer:object,report_every:int = None,tolerance:float = 1e-5):
        """
        Batch Size is left to be chosen when the Optimizer is initialized.
        learning_rate can be a function that changes based on certain conditions(TBD) or it can be a constant.
        """
        if not callable(learning_rate):
            if not isinstance(learning_rate,(int,float)):
                raise TypeError('learning_rate must be a function or a number')
            tmp = learning_rate
            learning_rate = lambda _: tmp
        
        for epoch in range(epochs):
            Optimizer.step(learning_rate(epoch)) #updates params by one step. If using BGD, updates params once. If using SGD, updates params len(X) times. If using MiniBatch, updates params num_batches times.
            self.epoch += 1
            
            if report_every is not None and self.epoch%report_every==0:
                report = [self.epoch,self.Loss(self = self.CostFunction,NeuralNetwork = self,Inputs = self.X,Targets = self.Y,update = True)]
                grad = self.dLoss(self = self.CostFunction,NeuralNetwork=self,Inputs = self.X,Targets = self.Y,update = False).mean(axis = 0)
                report.append(np.sqrt((grad*grad).sum()))

                if report[1] < tolerance:
                    print(f'Epoch: {self.epoch} Loss: {report[1]} Gradient Norm: {report[2]}')
                    print(f'Loss is less than tolerance of {tolerance}. Stopping training.')
                    return None
                print(f'Epoch: {self.epoch} Loss: {report[1]} Gradient Norm: {report[2]}',end = '\r')
                report = np.array([report],dtype = 'object')
                self.Epoch_Loss_Grad = np.append(self.Epoch_Loss_Grad,report,axis = 0)



    def Predict(self,Inputs:np.ndarray = None):
        if self.epoch == 0:
            raise ValueError('The Neural Network has not been trained yet.')
        '''Predicts the Targets of the Inputs'''
        if Inputs is None:
            return self.Tscales.unscale(self.forward(self.X))
        return self.Tscales.unscale(self.forward(self.Iscales.scale(Inputs)))
    
    def Predict_to_pandas(self,input_names:list[str],target_names:list[str],Inputs:np.ndarray = None,Targets:np.ndarray = None):
        '''returns a pandas datafrane of the inputs and targets along with their predictions'''
        if Inputs is None:
            Inputs = self.Inputs
        if Targets is None:
            Targets = self.Targets
        Predictions = self.Predict(Inputs)

        indf = pd.DataFrame(Inputs.reshape(-1,self.input_size),columns = input_names)
        tardf = pd.DataFrame(Targets.reshape(-1,self.output_size),columns = [target_name+'_t' for target_name in target_names])
        preddf = pd.DataFrame(Predictions.reshape(-1,self.output_size),columns = [target_name+'_p' for target_name in target_names])

        return pd.concat([indf,tardf,preddf],axis = 1)
    
    def showTraining(self,figsize = (16,9)):
        # Plot epoch versus loss and epoch versus gradient
        fig, ax1 = plt.subplots(figsize=figsize)

        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(self.Epoch_Loss_Grad[1:,0], self.Epoch_Loss_Grad[1:,1], color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        # Create a second y-axis for the gradient
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Gradient', color=color)
        ax2.plot(self.Epoch_Loss_Grad[1:,0], self.Epoch_Loss_Grad[1:,2], color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        plt.title('Epoch versus Loss and Gradient')
        plt.show()
        
    # For Saving and/or Loading
    def save_xlsx(self,filepath):
        Params,layer = {},0
        for weights, biases in zip(self.W.values(),self.B.values()):
            Wdf = pd.DataFrame(weights.copy())
            Bdf = pd.DataFrame(biases.copy())

            Params[f'Weights Layer {layer}'] = Wdf
            Params[f'Biases Layer {layer}' ] = Bdf
            layer+=1

        with pd.ExcelWriter(filepath) as writer:
            for sheet_name,df in Params.items():
                df.to_excel(writer,sheet_name = sheet_name, index = False)

        return f'successfully saved params to {filepath}'
    # Vizualization
    def CreateGraph(self,imagepath  = None):
        print('Hello')
        # Define the normalize function
        def normalize_values(values):
            min_val = min(values)
            max_val = max(values)

            if min_val == max_val:
                return [0.5 for _ in values]
            return [(val - min_val) / (max_val - min_val) for val in values]

        # Normalize weights and biases
        normalized_weights = np.concatenate([normalize_values(np.abs(self.W[i].copy()).flatten()) for i in range(1, len(self.hidden_sizes) + 2)])
        normalized_biases = np.concatenate([normalize_values(np.abs(self.B[i].copy()).flatten()) for i in range(1, len(self.hidden_sizes) + 2)])

        # Define the architecture of the neural network
        input_size = self.input_size
        output_size = self.output_size

        # Create a directed graph
        G = nx.DiGraph()

        # Add nodes for input layer
        input_nodes = ['Input{}'.format(i) for i in range(input_size)]
        G.add_nodes_from(input_nodes, layer=0)
        layers = [input_nodes]

        # Add nodes from hidden layers
        layer = 1
        for hidden_size in self.hidden_sizes:
            layers.append([f'Hidden {layer},'+'{}'.format(i) for i in range(hidden_size)])
            G.add_nodes_from(layers[-1],layer = layer)
            layer +=1

        output_nodes = ['Output{}'.format(i) for i in range(output_size)]
        G.add_nodes_from(output_nodes, layer=3)
        layers.append(output_nodes)

        for lag_layer,layer in zip(layers[:-1],layers[1:]):
            for i in lag_layer:
                for j in layer:
                    G.add_edge(i,j)

        # Position nodes
        pos = {}
        sizes = [input_size] + self.hidden_sizes + [output_size]
        x = 0

        for size,layer in zip(sizes,layers):
            num_nodes = len(layer)
            y_offset = (max(sizes) - num_nodes) // 2

            for i in range(num_nodes):
                pos[layer[i]] = (x, i + y_offset)
            x+=1

        # Draw the network
        plt.figure(figsize=(32, 18),facecolor = 'white')

        # Define colormap for linewidth and node color
        colormap = plt.cm.get_cmap('gray_r')

        for edge in G.edges():
            i, j = edge
            weight_norm = normalized_weights[int(j[-1]) - 1]
            linewidth = 0.1 + 2 * weight_norm

            nx.draw_networkx_edges(G, pos, edgelist=[(i, j)], width=linewidth, edge_color=colormap(weight_norm))

        for node in G.nodes():
            layer = G.nodes[node]['layer']
            if layer == 0:
                node_size = 100
                node_color = 'red'
            else:
                bias_norm = normalized_biases[int(node[-1]) - 1]
                node_size = 10 + 100 * bias_norm
                node_color = 'skyblue'

            nx.draw_networkx_nodes(G, pos, nodelist=[node], node_size=node_size, node_color=node_color)

        plt.axis('off')

        # Add titles and show the plot
        plt.title('Feedforward Neural Network')
        plt.show()
        if imagepath is not None:
            plt.savefig(imagepath, bbox_inches='tight', dpi=300)