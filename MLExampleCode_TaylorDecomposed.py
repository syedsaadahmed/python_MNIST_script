#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import matplotlib.pyplot as plt

# # Input data
X = np.array([0.0,0.0,0.0])


# In[15]:


# Simple neural-network based regressor
class NeuralNetwork:
# Function called at object initialization
    def __init__(self):

        # These are members of the class. We can access them in every method by \"self.var_name\" 
        #and from outside the class with \"instance_name.var_name\"

        # Sample to compute pass with # Inputs
        self.X = 0.0 # 1st input # will be input by user 
        self.y = 0.0 # set 2nd input # will be input by user
        self.z = 0.0 # set 3rd input # will be input by user
        
        self.X_new = 0.0 # new 1st input # will be calculated 
        self.y_new = 0.0 # new 2nd input # will be calculated
        self.z_new = 0.0 # new 2nd input # will be calculated
        
        self.goalOutput = 0.0 # set goalOutput # will be input by user
        
        
        # Distribution percentages
        self.PL1_N1 = 0.688 # % to distribut main output (L1) to newron_1 (N1) of prev layer 
        self.PL1_N2 = 0.312 # % to distribut main output (L1) to newron_2 (N2) of prev layer
        
        self.PN1_I1 = 0.366 # % to distribut newron_1 to Input 1 
        self.PN1_I2 = 0.282 # % to distribut newron_1 to Input 2
        self.PN1_I3 = 0.352 # % to distribut newron_1 to Input 3
        
        self.PN2_I1 = 0.333 # % to distribut newron_2 to Input 1 
        self.PN2_I2 = 0.417 # % to distribut newron_2 to Input 2
        self.PN2_I3 = 0.250 # % to distribut newron_2 to Input 3
                
        # weights + biases 1st layer 1st neuron
        self.weight_11   = 0.4 # weight for 1st input 1st neuron
        self.bias_11   = 0.2 # bias for 1st input 1st neuron

        self.weight_21   = 0.7 # weight for 2nd input 1st neuron
        self.bias_21   = 0.3 # bias for 2nd input 1st neuron

        self.weight_31   = 0.9 # weight for 3rd input 1st neuron
        self.bias_31   = 0.1 # bias for 3rd input 1st neuron

        # weights + biases 1st layer 2nd neuron
        self.weight_12   = 0.1 # weight for 1st input 2nd neuron
        self.bias_12   = 0.05 # bias for 1st input 2nd neuron

        self.weight_22   = 0.5 # weight for 2nd input 2nd neuron
        self.bias_22   = 0.1 # bias for 2nd input 2nd neuron

        self.weight_32   = 0.2 # weight for 3rd input 2nd neuron
        self.bias_32   = 0.01 # bias for 3rd input 2nd neuron

        # weights + biases of final layer 1st neuron
        self.weight_l21   = 0.5 # weight for 2nd layer 1st neuron
        self.bias_l21   = 0.3 # bias for 2nd layer 1st neuron

        # weights + biases of final layer 2nd neuron
        self.weight_l22  = 0.7 # weight for 2nd layer 2nd neuron
        self.bias_l22   = 0.2 # bias for 2nd layer 2nd neuron

        # State information
        self.out_hidden_1     = 0.0 # output for 1st layer 1st neuron
        self.out_hidden_1_new = 0.0 # new output for 1st layer 1st neuron
        
        self.out_hidden_2     = 0.0 # output for 1st layer 2nd neuron
        self.out_hidden_2_new = 0.0 # new output for 1st layer 2nd neuron

        self.output     = 0.0 # Feedforward Final Output # will be calculated

    # Set sample to be used in feed-forward and back-propagation pass
    def set_sample(self, X, y, z, goalOutput):
        self.X = float(X)
        self.y = float(y)
        self.z = float(z)
        self.goalOutput = float(goalOutput)

    ## Feed-forward pass
    def feed_forward(self):
        net_hidden_11 = (self.X * self.weight_11) + self.bias_11
        net_hidden_21 = (self.y * self.weight_21) + self.bias_21
        net_hidden_31 = (self.z * self.weight_31) + self.bias_31
        self.out_hidden_1 = ReLu(net_hidden_11+net_hidden_21+net_hidden_31) #ReLu function // output 1st neuron of Hidden layer 1
        print("Value of Hidden Neuron_1: " + str(round(self.out_hidden_1,3)))
        
        net_hidden_12 = (self.X * self.weight_12) + self.bias_12
        net_hidden_22 = (self.y * self.weight_22) + self.bias_22
        net_hidden_32 = (self.z * self.weight_32) + self.bias_32
        self.out_hidden_2 = ReLu(net_hidden_12+net_hidden_22+net_hidden_32) #ReLu function // output 2nd neuron of Hidden layer 1
        print("Value of Hidden Neuron_2: " + str(round(self.out_hidden_2,3)))

        net_out_1 = (self.out_hidden_1  * self.weight_l21) + self.bias_l21
        net_out_2 = (self.out_hidden_2  * self.weight_l22) + self.bias_l22
        
#         print("net_out_1 : " + str(net_out_1))
#         print("net_out_2 : " + str(net_out_2))

        self.output = sigmoid(net_out_1+net_out_2) #sigmoid function // Predicted Output

    ## Back-propagation for Taylor Decomposition and Find new Input Values
    def back_prop(self):
        self.out_hidden_1_new = self.out_hidden_1 - (((self.output-self.goalOutput)*self.PL1_N1) / (self.goalOutput * (1-self.goalOutput) * self.weight_l21))        
        self.out_hidden_2_new = self.out_hidden_2 - (((self.output-self.goalOutput)*self.PL1_N2) / (self.goalOutput * (1-self.goalOutput) * self.weight_l22))
                
        if(self.out_hidden_1<0 or self.out_hidden_2<0):
            return false
        else:
            X_new_a = self.X - (((self.out_hidden_1 - self.out_hidden_1_new) * self.PN1_I1) / (self.weight_11))
            y_new_a = self.y - (((self.out_hidden_1 - self.out_hidden_1_new) * self.PN1_I2) / (self.weight_21))
            z_new_a = self.z - (((self.out_hidden_1 - self.out_hidden_1_new) * self.PN1_I3) / (self.weight_31))

            X_new_b = self.X - (((self.out_hidden_2 - self.out_hidden_2_new) * self.PN2_I1) / (self.weight_12))
            y_new_b = self.y - (((self.out_hidden_2 - self.out_hidden_2_new) * self.PN2_I2) / (self.weight_22))
            z_new_b = self.z - (((self.out_hidden_2 - self.out_hidden_2_new) * self.PN2_I3) / (self.weight_32))

            self.X_new = X_new_a + X_new_b 
            self.y_new = y_new_a + y_new_b
            self.z_new = z_new_a + z_new_b

            return true
        
# Sigmoid Function
def sigmoid(s):
    return (1/(1+np.exp(-s)))

# ReLu Function
def ReLu(s):
    return (max(0,s))

def execute_nn(X, y, z, goalOutput):
    nn = NeuralNetwork() # Instantiate neural network
    nn.set_sample(X, y, z, goalOutput) # set input values
    
    isDecomposable = nn.feed_forward() # perform feed-forward to calculate output        
    print("\nFinal Output: "+ str(round(nn.output,3)) + "\n")
    
    if(isDecomposable):
        print(" *** Applying Tylor Decomposition *** \n")
        nn.back_prop()

        print("New Value of Hidden Neuron 1: "+ str(round(nn.out_hidden_1_new,3)))
        print("New Value of Hidden Neuron 2: "+ str(round(nn.out_hidden_2_new,3)) + "\n")
    #     print("New Value of Hidden Neuron 2: "+ str(round(nn.out_hidden_2_new,3)) + "\n")
    
        print("New first Input: "+ str(round(nn.X_new,3)))
        print("New 2nd Input: "+ str(round(nn.y_new,3)))
        print("New 3rd Input: "+ str(round(nn.z_new,3)))
    else:
        print("-ve values encountered for ReLU. Please select different % Distributions")

print(" *** Initial Input values *** \n 0.7, \n 0.1, \n 0.4 \n")
execute_nn(0.7,0.1, 0.4, 0.65)
# execute_nn(-0.448,-0.423, 0.062, 0.65)


# In[ ]:




