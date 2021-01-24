import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# # Input data
X = np.array([0.0,0.0,0.0])

class NeuralNetwork:
    def __init__(self):
        self.result = pd.DataFrame(columns=['Inputs', 'P1', 'P2' , 'Inputs_New', "temp_output" , 'Difference', "Euc_Distance"])
        self.distance = 0.000
        # These are members of the class. We can access them in every method by \"self.var_name\" 
        #and from outside the class with \"instance_name.var_name\"

        self.inputs = {"X": 0.000, "y": 0.000, "z": 0.000} # input # will be input by user 
        self.inputs_new = {"X": 0.000, "y": 0.000, "z": 0.000} # new input # will be calculated
        self.inputs_diff = {"X": 0.000, "y": 0.000, "z": 0.000} # difference # will be calculated

        self.goalOutput = 0.000 # set goalOutput # will be input by user
        
        # Neuron Backward Distribution percentages
        self.PL1_N1 = 0.000 # % to distribut main output (L1) to newron_1 (N1) of prev layer 
        self.PL1_N2 = 0.000 # % to distribut main output (L1) to newron_2 (N2) of prev layer
        
        self.PN1_I1 = 0.366 # % to distribut newron_1 to Input 1 
        self.PN1_I2 = 0.282 # % to distribut newron_1 to Input 2
        self.PN1_I3 = 0.352 # % to distribut newron_1 to Input 3
        
        self.PN2_I1 = 0.333 # % to distribut newron_2 to Input 1 
        self.PN2_I2 = 0.417 # % to distribut newron_2 to Input 2
        self.PN2_I3 = 0.250 # % to distribut newron_2 to Input 3
                
        # weights + biases 1st layer 1st neuron
        self.weight_11   = 0.400 # weight for 1st input 1st neuron
        self.bias_11   = 0.200 # bias for 1st input 1st neuron

        self.weight_21   = 0.700 # weight for 2nd input 1st neuron
        self.bias_21   = 0.300 # bias for 2nd input 1st neuron

        self.weight_31   = 0.900 # weight for 3rd input 1st neuron
        self.bias_31   = 0.100 # bias for 3rd input 1st neuron

        # weights + biases 1st layer 2nd neuron
        self.weight_12   = 0.100 # weight for 1st input 2nd neuron
        self.bias_12   = 0.050 # bias for 1st input 2nd neuron

        self.weight_22   = 0.500 # weight for 2nd input 2nd neuron
        self.bias_22   = 0.100 # bias for 2nd input 2nd neuron

        self.weight_32   = 0.200 # weight for 3rd input 2nd neuron
        self.bias_32   = 0.010 # bias for 3rd input 2nd neuron

        # weights + biases of final layer 1st neuron
        self.weight_l21   = 0.500 # weight for 2nd layer 1st neuron
        self.bias_l21   = 0.300 # bias for 2nd layer 1st neuron

        # weights + biases of final layer 2nd neuron
        self.weight_l22  = 0.700 # weight for 2nd layer 2nd neuron
        self.bias_l22   = 0.200 # bias for 2nd layer 2nd neuron

        # State information
        self.out_hidden_1     = 0.000 # output for 1st layer 1st neuron
        self.out_hidden_1_new = 0.000 # new output for 1st layer 1st neuron
        
        self.out_hidden_2     = 0.000 # output for 1st layer 2nd neuron
        self.out_hidden_2_new = 0.000 # new output for 1st layer 2nd neuron

        self.output     = 0.000 # Feedforward Final Output # will be calculated
        self.xoutput = 0.000 # Feedforward Temp Output # will be calculated

    # Set sample to be used in feed-forward and back-propagation pass
    def set_sample(self, X, y, z, goalOutput):
        self.inputs["X"] = X #float(X)
        self.inputs["y"] = y #float(y)
        self.inputs["z"] = z #float(z)
        self.goalOutput = goalOutput #float(goalOutput)

    def setDistributionPercentages(self, p1, p2):
#         self.PL1_N1 = round(float(p1),3) # % to distribut main output (L1) to newron_1 (N1) of prev layer 
#         self.PL1_N2 = round(float(p2),3) # % to distribut main output (L1) to newron_2 (N2) of prev layer
        self.PL1_N1 = p1 #float(p1) # % to distribut main output (L1) to newron_1 (N1) of prev layer 
        self.PL1_N2 = p2 #float(p2) # % to distribut main output (L1) to newron_2 (N2) of prev layer

    ## Feed-forward pass
    def feed_forward(self):
        net_hidden_11 = (self.inputs["X"] * self.weight_11) + self.bias_11
        net_hidden_21 = (self.inputs["y"] * self.weight_21) + self.bias_21
        net_hidden_31 = (self.inputs["z"] * self.weight_31) + self.bias_31
        self.out_hidden_1 = ReLu(net_hidden_11+net_hidden_21+net_hidden_31) #ReLu function // output 1st neuron of Hidden layer 1
        print("Value of Hidden Neuron_1: " + str(round(self.out_hidden_1,3)))
        
        net_hidden_12 = (self.inputs["X"] * self.weight_12) + self.bias_12
        net_hidden_22 = (self.inputs["y"] * self.weight_22) + self.bias_22
        net_hidden_32 = (self.inputs["z"] * self.weight_32) + self.bias_32
        self.out_hidden_2 = ReLu(net_hidden_12+net_hidden_22+net_hidden_32) #ReLu function // output 2nd neuron of Hidden layer 1
        print("Value of Hidden Neuron_2: " + str(round(self.out_hidden_2,3)))

        net_out_1 = (self.out_hidden_1  * self.weight_l21) + self.bias_l21
        net_out_2 = (self.out_hidden_2  * self.weight_l22) + self.bias_l22
        
        self.output = round(sigmoid(net_out_1+net_out_2),3) #sigmoid function // Predicted Output
        
    ## Temp Feed-forward pass
    def temp_feed_forward(self):
        xnet_hidden_11 = (self.inputs_new["X"] * self.weight_11) + self.bias_11
        xnet_hidden_21 = (self.inputs_new["y"] * self.weight_21) + self.bias_21
        xnet_hidden_31 = (self.inputs_new["z"] * self.weight_31) + self.bias_31
        xout_hidden_1 = ReLu(xnet_hidden_11+xnet_hidden_21+xnet_hidden_31) #ReLu function // output 1st neuron of Hidden layer 1
#         print("Value of Hidden Neuron_1: " + str(round(xout_hidden_1,3)))
        
        xnet_hidden_12 = (self.inputs_new["X"] * self.weight_12) + self.bias_12
        xnet_hidden_22 = (self.inputs_new["y"] * self.weight_22) + self.bias_22
        xnet_hidden_32 = (self.inputs_new["z"] * self.weight_32) + self.bias_32
        xout_hidden_2 = ReLu(xnet_hidden_12+xnet_hidden_22+xnet_hidden_32) #ReLu function // output 2nd neuron of Hidden layer 1
#         print("Value of Hidden Neuron_2: " + str(round(xout_hidden_2,3)))

        xnet_out_1 = (xout_hidden_1  * self.weight_l21) + self.bias_l21
        xnet_out_2 = (xout_hidden_2  * self.weight_l22) + self.bias_l22
        
        self.xoutput = round(sigmoid(xnet_out_1+xnet_out_2),4) #sigmoid function // Predicted Output

    ## Back-propagation for Taylor Decomposition and Find new Input Values
    def back_prop(self):
        self.out_hidden_1_new = self.out_hidden_1 - (((self.output-self.goalOutput)*self.PL1_N1) / (self.goalOutput * (1-self.goalOutput) * self.weight_l21))        
        self.out_hidden_2_new = self.out_hidden_2 - (((self.output-self.goalOutput)*self.PL1_N2) / (self.goalOutput * (1-self.goalOutput) * self.weight_l22))
                
        if(self.out_hidden_1<0 or self.out_hidden_2<0):
            print("-ve values encountered for ReLU. Please select different % Distributions")
            self.result = self.result.append({'Inputs': self.inputs.values(), 'P1': str(self.PL1_N1) + "%", 'P2': str(self.PL1_N2)+ "%", 'Inputs_New': "x x x", "temp_output" : "N/A", 'Difference': "x x x", "Euc_Distance" : "N/A"}, ignore_index=True)
#             row = [self.inputs.values(), str(self.PL1_N1) + "%", str(self.PL1_N2)+ "%", listToString(["x","x","x"]), listToString(["x","x","x"])]
#             self.result.loc[len(self.result)] = row
            return False
        else:
            X_new_a = self.inputs["X"] - (((self.out_hidden_1 - self.out_hidden_1_new) * self.PN1_I1) / (self.weight_11))
            y_new_a = self.inputs["y"] - (((self.out_hidden_1 - self.out_hidden_1_new) * self.PN1_I2) / (self.weight_21))
            z_new_a = self.inputs["z"] - (((self.out_hidden_1 - self.out_hidden_1_new) * self.PN1_I3) / (self.weight_31))

            X_new_b = self.inputs["X"] - (((self.out_hidden_2 - self.out_hidden_2_new) * self.PN2_I1) / (self.weight_12))
            y_new_b = self.inputs["y"] - (((self.out_hidden_2 - self.out_hidden_2_new) * self.PN2_I2) / (self.weight_22))
            z_new_b = self.inputs["z"] - (((self.out_hidden_2 - self.out_hidden_2_new) * self.PN2_I3) / (self.weight_32))

            self.inputs_new["X"] = round(X_new_a + X_new_b,3)
            self.inputs_new["y"] = round(y_new_a + y_new_b,3)
            self.inputs_new["z"] = round(z_new_a + z_new_b,3)

            self.inputs_diff["X"] = round(self.inputs["X"] - self.inputs_new["X"],3)
            self.inputs_diff["y"] = round(self.inputs["y"] - self.inputs_new["y"],3)
            self.inputs_diff["z"] = round(self.inputs["z"] - self.inputs_new["z"],3)

            point1 = np.fromiter(self.inputs.values(), dtype=float)
            point2 = np.fromiter(self.inputs_new.values(), dtype=float)

            self.distance = np.linalg.norm(point1 - point2)
            self.result = self.result.append({'Inputs': self.inputs.values(), 'P1': str(self.PL1_N1) + "%", 'P2': str(self.PL1_N2)+ "%", 'Inputs_New': listToString(self.inputs_new.values()), "temp_output" : self.xoutput, 'Difference': listToString(self.inputs_diff.values()),"Euc_Distance" : self.distance}, ignore_index=True)            
#             row = [self.inputs.values(), str(self.PL1_N1) + "%", str(self.PL1_N2)+ "%", listToString(self.inputs_new.values()), listToString(self.inputs_diff.values())]
#             self.result.loc[len(self.result)] = row
            return True

def listToString(s):
    str1 = " "
    for ele in s:  
        str1 += str(ele) + " " 
    return str1 

# Sigmoid Function
def sigmoid(s):
    return (1/(1+np.exp(-s)))

# ReLu Function
def ReLu(s):
    return (max(0,s))

def execute_nn(X, y, z, goalOutput):
    nn = NeuralNetwork() # Instantiate neural network
    print(" *** Feed Forward *** \n")
    nn.set_sample(X, y, z, goalOutput) # set input values
    nn.setDistributionPercentages(0.688,0.312) # 0.688,0.312
    nn.feed_forward() # perform feed-forward to calculate output        
    print("\nFinal Output: "+ str(round(nn.output,3)) + "\n")
    
    print(" *** Applying Tylor Decomposition *** Turn " + str(nn.output) + " into " + str(nn.goalOutput) +" \n")
    
    P1=0.99
    P2=0.01
    
    for x in range(99):
        nn.setDistributionPercentages(P1,P2) # 0.688,0.312 # % to distribut main output (L1) to newrons of prev layer
        isDecomposable = nn.back_prop()
        nn.temp_feed_forward() # calculate temp feed forward on new inputs
        
        P1=round(P1 - 0.01,3)
        P2=round(P2 + 0.01,3)
    print(nn.result.to_string())
# print("Initial Input values \t 0.7, \t 0.1, \t 0.4 \t")
execute_nn(0.7,0.1, 0.4, 0.65)
# execute_nn(0.190925, -0.326, 0.2849, 0.65)