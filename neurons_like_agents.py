import numpy as np
import random


a=1.1 #parameter for not too strict constraints for maximum of utility function
b=0.2 #speed of moving to the target node
c=0.1 #decay of energy in the node



#Initial lists
list_of_axons=[]

list_of_dendrites=[]

list_of_active_dendrites=[]


#MSE Function of Joy. Utility function can be anything in general

def mse_joy (y, y_pred):
    
    mse_j = -1*(np.mean((y - y_pred)**2))
    
    return mse_j

UF_max=-1000000000 #Initial very big negative utility function value for comparison


class Axon():
    
    '''
    Axon class for passing signals
    '''
    
    def __init__(self, initial_coordinates, neuron):
        
        self.coordinates=initial_coordinates
        self.parent_neuron=neuron
        self.axon_sparkling=False
    
    def signal_of_axon (self):
        
        neuron_logits=self.parent_neuron.neuron_proceed()
        self.axon_sparkling=True
        
        return neuron_logits
    
    def check_for_sparkling(self):
        
        #We need this function for sampling possibilities for connections in the current node
        
        neuron_logits=self.parent_neuron.neuron_proceed()
        print(f'signal in axon: {neuron_logits}')
        if np.sum(neuron_logits)!=0:
        #if np.abs(neuron_logits) > 1:
            self.axon_sparkling=True
        else:
            print(f'print little signal: {neuron_logits}')
            self.axon_sparkling=False
        
        

class Neuron():
    
    '''
    Neuron sums all inputs and implement some filters. But this is huge field of researches
    '''
    
    def __init__(self, dendrites: list):
        
        self.child_dendrites_list=dendrites
        
    
    def neuron_proceed(self):
        
        '''
        Main computing function of the neuron
        '''
        
        all_together=0
        for dendrite in self.child_dendrites_list:
            dendrite_logits=dendrite.dendrite_proceed()
            #print(f'dendrite signal in neuron: {dendrite.obtained_signal}')
            #print(f'dendrite logits: {dendrite_logits}')
            all_together+=dendrite_logits
        
        ''' Maybe here we should eliminate influence of the connection after proceeding. But in later versions'''
        #neuron_logits=np.maximum(all_together,0.0)
        neuron_logits=all_together
        
            
        return neuron_logits

class Dendrite():
    
    '''
    Dendrite receives signal and produces changes to it.
    '''
    
    def __init__(self,initial_coord):
        
        self.coord=initial_coord
        #self.parent_neuron=neuron
        self.UF=0
        self.UF_local=-10**8
        self.obtained_signal=0
        self.connection_established=False
        
        #self.not_this_timestamp_connection=True
    
    def dendrite_proceed(self):
        
        ''' We need more sophisticated temporal self-dependent function. But not necessary. Could be anything'''
        
        
        return self.obtained_signal*1.1
        
    
    
    def moving(self):
        
        '''Dendrite searches the most attractive direction and makes a move'''
        
        list_of_attraction=[]
        
        list_of_gravity=[]
        
        #compute distances for each node
        for i in range(len(list_of_nodes)):
            
            list_of_quasi_dist=[]
            
            for j in range(len(self.coord)):
                dist=(self.coord[j]-list_of_nodes[i].coordinates[j])
                
                #print(dist)
                
                distance=dist*dist
                list_of_quasi_dist.append(distance)
                
            big_distance=sum(list_of_quasi_dist)
            
            #print(big_distance)
            
            #Conditional distance taking into accounts energy of the node
            gravity_distance=list_of_nodes[i].node_energy**2/(big_distance+0.00001)
            
            print(f' node energy: {list_of_nodes[i].node_energy}')
            
            list_of_attraction.append(gravity_distance)
            
            #print(list_of_attraction)
            
        
        
    
        #compute quasi-probabilities from inverses of distances
        for k in range(len(list_of_attraction)):
            
            #eliminating 100% probability of beign in the current node and nodes without sparkling axons
           
            
            if (list_of_attraction[k]==0 or len(list_of_nodes[k].list_of_sparkling_axons)==0): #or len(list_of_nodes[k].list_of_sparkling_axons)==0:
                gravity=0
            
            else:
                gravity=list_of_attraction[k]
                
            all_gravity_force=sum(list_of_attraction)
            true_gravity=gravity/all_gravity_force
            list_of_gravity.append(true_gravity)
        
        #Define target node
        target_node_index=np.argmax(list_of_gravity)
        target_coord=list_of_nodes[target_node_index]
        print(f'target node index: {target_node_index} and node: {target_coord}')
        print(f'list of sparkling axon in the chosen node: {target_coord.list_of_sparkling_axons}')
        
        #Moving to the target node with some speed and updating coordinates of the dendrite
        new_coord=[]
        for c in range(len(self.coord)):
            new_coord_local=self.coord[c]+b*((target_coord.coordinates[c])-self.coord[c]) #b is exogeneous speed factor of moving dendrites
            new_coord.append(new_coord_local)
        
        self.coord=tuple(new_coord)
        print(f'new coordinates of dendrite: {self.coord}')
        
        
    
    def try_connection(self):
        
        '''Try to connect with sparkling axon'''
        
        
        #Revealing the closest node to our dendrite
        
        list_of_distances=[]
        
        for i in range(len(list_of_nodes)):
            
            list_of_quasi_dist=[]
            
            
            for j in range(len(self.coord)):
                dist=(self.coord[j]-list_of_nodes[i].coordinates[j])
                distance=dist*dist
                list_of_quasi_dist.append(distance)
            big_distance=sum(list_of_quasi_dist)
            
            #print(big_distance)
            
            list_of_distances.append(big_distance)
            
            #print(list_of_distances)
            
        #Finding node with minimal distanece to dendrite
        
        index_of_current_node=np.argmin(list_of_distances)
        
        print(f'index of the most attractive node: {index_of_current_node}')
        
        current_node=list_of_nodes[index_of_current_node]
        
        #Try connection
        
        if current_node.list_of_sparkling_axons !=[]: #Make sure, that sequense of the sparkling axons is not empty
            
            print(f'length of sparkling axons: {len(current_node.list_of_sparkling_axons)}')
            
            #Random choice, if there are many sparkling nodes in the node
            axon_try=random.choice(current_node.list_of_sparkling_axons) 
            
            #Transpassing of the signal
            self.obtained_signal=axon_try.signal_of_axon()
            logits=self.dendrite_proceed()
            
            print(f'local logits: {logits}')
            
            #Computing utility function
            self.UF_local=mse_joy(y=y,y_pred=logits)
            
            print(f'local utility function value: {self.UF_local}')
            
            #Compare and decide about connection
            if self.UF_local>a*UF_max:
                
                #print(f'length of sparkling axons: {current_node.list_of_sparkling_axons}')
                self.connection_established=True
                
                self.obtained_signal=axon_try.signal_of_axon()
                axon_try.axon_sparkling=False
                current_node.list_of_sparkling_axons.remove(axon_try)
                current_node.list_of_established_axons.append(axon_try)
                self.UF+=1
                print(f'Connection established and node is {current_node.list_of_established_axons}')
                
            else: self.obtained_signal=0.00
            
        else:
            print(f' Sequence is empty: {current_node.list_of_sparkling_axons}')
            
        print(f'internal signal is {self.obtained_signal}')
            

#Another class of dendrite with different internal function
class Dendrite_Lower():
    
    def __init__(self,initial_coord):
        
        self.coord=initial_coord
        #self.parent_neuron=neuron
        self.UF=0
        self.UF_local=-10**8
        self.obtained_signal=0
        self.connection_established=False
        
        #self.not_this_timestamp_connection=True
    
    def dendrite_proceed(self):
        
        ''' We need more sophisticated temporal self-dependent function. But not necessery. Could be anything'''
        
        
        #Different function
        return self.obtained_signal*0.9
        
    
    
    def moving(self):
        
        
        list_of_attraction=[]
        
        list_of_gravity=[]
        
        #compute distances for each node
        for i in range(len(list_of_nodes)):
            
            list_of_quasi_dist=[]
            
            for j in range(len(self.coord)):
                dist=(self.coord[j]-list_of_nodes[i].coordinates[j])
                
                #print(dist)
                
                distance=dist*dist
                list_of_quasi_dist.append(distance)
            big_distance=sum(list_of_quasi_dist)
            
            #print(big_distance)
            
            gravity_distance=list_of_nodes[i].node_energy**2/(big_distance+0.00001)
            
            print(f' node energy: {list_of_nodes[i].node_energy}')
            
            list_of_attraction.append(gravity_distance)
            
            #print(list_of_attraction)
            
        
        
    
        #compute quasi-probabilities from inverses of distances
        for k in range(len(list_of_attraction)):
            
            #eliminating 100% probability of beign in the current node and nodes without sparkling axons
            
            if (list_of_attraction[k]==0 or len(list_of_nodes[k].list_of_sparkling_axons)==0): #or len(list_of_nodes[k].list_of_sparkling_axons)==0:
                gravity=0
            
            else:
                gravity=list_of_attraction[k]
                
            all_gravity_force=sum(list_of_attraction)
            true_gravity=gravity/all_gravity_force
            list_of_gravity.append(true_gravity)
        
        target_node_index=np.argmax(list_of_gravity)
        target_coord=list_of_nodes[target_node_index]
        print(f'target node index: {target_node_index} and node: {target_coord}')
        print(f'list of sparkling axon in the chosen node: {target_coord.list_of_sparkling_axons}')
        
        new_coord=[]
        for c in range(len(self.coord)):
            new_coord_local=self.coord[c]+b*((target_coord.coordinates[c])-self.coord[c]) #b is exogeneous speed factor of moving dendrites
            new_coord.append(new_coord_local)
        
        self.coord=tuple(new_coord)
        print(f'new coordinates of dendrite: {self.coord}')
        
        
    
    def try_connection(self):
        
        list_of_distances=[]
        
        for i in range(len(list_of_nodes)):
            
            list_of_quasi_dist=[]
            
            for j in range(len(self.coord)):
                dist=(self.coord[j]-list_of_nodes[i].coordinates[j])
                distance=dist*dist
                list_of_quasi_dist.append(distance)
            big_distance=sum(list_of_quasi_dist)
            
            #print(big_distance)
            
            list_of_distances.append(big_distance)
            
            #print(list_of_distances)
            
        
        index_of_current_node=np.argmin(list_of_distances)
        
        print(f'index of the most attractive node: {index_of_current_node}')
        
        current_node=list_of_nodes[index_of_current_node]
        
        if current_node.list_of_sparkling_axons !=[]: #Make sure, that sequense of the sparkling axons is not empty
            
            print(f'length of sparkling axons: {len(current_node.list_of_sparkling_axons)}')
            
            axon_try=random.choice(current_node.list_of_sparkling_axons) 
            
            self.obtained_signal=axon_try.signal_of_axon()
            logits=self.dendrite_proceed()
            
            print(f'local logits: {logits}')
            
            self.UF_local=mse_joy(y=y,y_pred=logits)
            
            print(f'local utility function value: {self.UF_local}')
            
            if self.UF_local>a*UF_max:
                
                #print(f'length of sparkling axons: {current_node.list_of_sparkling_axons}')
                self.connection_established=True
                
                self.obtained_signal=axon_try.signal_of_axon()
                axon_try.axon_sparkling=False
                current_node.list_of_sparkling_axons.remove(axon_try)
                current_node.list_of_established_axons.append(axon_try)
                self.UF+=1
                print(f'Connection established and node is {current_node.list_of_established_axons}')
                
            else: self.obtained_signal=0.00
            
        else:
            print(f' Sequence is empty: {current_node.list_of_sparkling_axons}')
            
        print(f'internal signal is {self.obtained_signal}')
            

#And another function
class Dendrite_Minus():
    
    def __init__(self,initial_coord):
        
        self.coord=initial_coord
        #self.parent_neuron=neuron
        self.UF=0
        self.UF_local=-10**8
        self.obtained_signal=0
        self.connection_established=False
        
        #self.not_this_timestamp_connection=True
    
    def dendrite_proceed(self):
        
        ''' We need more sophisticated temporal self-dependent function. But not necessery. Could be anything'''
        
        
        #Another function
        return self.obtained_signal*(-1)
        
    
    
    def moving(self):
        
        
        list_of_attraction=[]
        
        list_of_gravity=[]
        
        #compute distances for each node
        for i in range(len(list_of_nodes)):
            
            list_of_quasi_dist=[]
            
            for j in range(len(self.coord)):
                dist=(self.coord[j]-list_of_nodes[i].coordinates[j])
                
                #print(dist)
                
                distance=dist*dist
                list_of_quasi_dist.append(distance)
            big_distance=sum(list_of_quasi_dist)
            
            #print(big_distance)
            
            gravity_distance=list_of_nodes[i].node_energy**2/(big_distance+0.00001)
            
            print(f' node energy: {list_of_nodes[i].node_energy}')
            
            list_of_attraction.append(gravity_distance)
            
            #print(list_of_attraction)
            
        
        
    
        #compute quasi-probabilities from inverses of distances
        for k in range(len(list_of_attraction)):
            
            #eliminating 100% probability of beign in the current node and nodes without sparkling axons
            
            if (list_of_attraction[k]==0 or len(list_of_nodes[k].list_of_sparkling_axons)==0): # or len(list_of_nodes[k].list_of_sparkling_axons)==0:
                gravity=0
            
            else:
                gravity=list_of_attraction[k]
                
            all_gravity_force=sum(list_of_attraction)
            true_gravity=gravity/all_gravity_force
            list_of_gravity.append(true_gravity)
        
        target_node_index=np.argmax(list_of_gravity)
        target_coord=list_of_nodes[target_node_index]
        print(f'target node index: {target_node_index} and node: {target_coord}')
        print(f'list of sparkling axon in the chosen node: {target_coord.list_of_sparkling_axons}')
        
        new_coord=[]
        for c in range(len(self.coord)):
            new_coord_local=self.coord[c]+b*((target_coord.coordinates[c])-self.coord[c]) #b is exogeneous speed factor of moving dendrites
            new_coord.append(new_coord_local)
        
        self.coord=tuple(new_coord)
        print(f'new coordinates of dendrite: {self.coord}')
        
        
    
    def try_connection(self):
        
        list_of_distances=[]
        
        for i in range(len(list_of_nodes)):
            
            list_of_quasi_dist=[]
            
            for j in range(len(self.coord)):
                dist=(self.coord[j]-list_of_nodes[i].coordinates[j])
                distance=dist*dist
                list_of_quasi_dist.append(distance)
            big_distance=sum(list_of_quasi_dist)
            
            #print(big_distance)
            
            list_of_distances.append(big_distance)
            
            #print(list_of_distances)
            
        
        index_of_current_node=np.argmin(list_of_distances)
        
        print(f'index of the most attractive node: {index_of_current_node}')
        
        current_node=list_of_nodes[index_of_current_node]
        
        if current_node.list_of_sparkling_axons !=[]: #Make sure, that sequense of the sparkling axons is not empty
            
            print(f'length of sparkling axons: {len(current_node.list_of_sparkling_axons)}')
            
            axon_try=random.choice(current_node.list_of_sparkling_axons) 
            
            self.obtained_signal=axon_try.signal_of_axon()
            logits=self.dendrite_proceed()
            
            print(f'local logits: {logits}')
            
            self.UF_local=mse_joy(y=y,y_pred=logits)
            
            print(f'local utility function value: {self.UF_local}')
            
            if self.UF_local>a*UF_max:
                
                #print(f'length of sparkling axons: {current_node.list_of_sparkling_axons}')
                self.connection_established=True
                
                self.obtained_signal=axon_try.signal_of_axon()
                axon_try.axon_sparkling=False
                current_node.list_of_sparkling_axons.remove(axon_try)
                current_node.list_of_established_axons.append(axon_try)
                self.UF+=1
                print(f'Connection established and node is {current_node.list_of_established_axons}')
                
            else: self.obtained_signal=0.00
            
        else:
            print(f' Sequence is empty: {current_node.list_of_sparkling_axons}')
            
        print(f'internal signal is {self.obtained_signal}')
            




class Node():
    
    '''
    Nodes are for aggregation of the axons. Need for computing simplification in the very big models
    '''
    
    def __init__(self, coordinates):
        
        self.coordinates=coordinates
        self.node_energy=1
        self.list_of_sparkling_axons=[]
        self.list_of_established_axons=[]
        self.energy_list=[]
    
    def energy_count(self):
        
        ''' Need to make timescale for counting energy'''
        
        self.node_energy=1
        for axon in self.list_of_established_axons:
            self.node_energy+=1
        
        if self.node_energy>1:
            self.node_energy=self.node_energy-c*self.energy_list[-1]
            self.energy_list.append(self.node_energy)
        else:
            self.node_energy=self.node_energy
            self.energy_list.append(self.node_energy)


'''
Method of counting sparkle axons in the node:

'''
def count_sparkling_axons (list_of_nodes, list_of_axons): 
    
    for node in list_of_nodes:
        
        for axon in list_of_axons:
            
            axon.check_for_sparkling()
            
            if axon.axon_sparkling == True:
                print("True")
                if axon not in node.list_of_established_axons:
                    if axon not in node.list_of_sparkling_axons:
                        if node.coordinates==axon.coordinates:
                            node.list_of_sparkling_axons.append(axon)
        print(f'local node sparkling axons list: {node.list_of_sparkling_axons}')


                   
#Method for count active dendrites: list of all dendrite --> list of dendrites if connection established=False

def count_active_dendrites (list_of_dendrites):
    
    for dendrite in list_of_dendrites:
        
        if dendrite not in list_of_active_dendrites:
            
            #print('Not in active list')
            
            if dendrite.connection_established == False:
                
                #print("Connection is not established")
            
                list_of_active_dendrites.append(dendrite)
            
    #print(f'list of active dendrites: {list_of_active_dendrites}')
            
#Method of removing estblished dendrites
def remove_from_active_dendrites ():
    
    
    print(f'len of the list_of_dendrites: {len(list_of_dendrites)}')
    print(f'len of the list_of_active_dendrites: {len(list_of_active_dendrites)}')
    
    for dendrite in list_of_dendrites:
         
        if dendrite.connection_established == True and dendrite in list_of_active_dendrites:
            list_of_active_dendrites.remove(dendrite)

Timestamps=10 #Exogeneous parameter
UFglobal=[]




            
    
'''

Need to initialize all dendrite, axons, neurons, nodes, x_data and all corresponding lists


Inintialization: Denderites --> Neuron --> Axons

'''
#First Neuron

first_neuron_dendrite=Dendrite(initial_coord=(0,0))

first_neuron_dendrite_list=[first_neuron_dendrite]
first_neuron=Neuron(dendrites=first_neuron_dendrite_list)

first_neuron_axon_1=Axon((2,2), neuron=first_neuron)
first_neuron_axon_2=Axon((2,0), neuron=first_neuron)
first_neuron_axon_3=Axon((2,3), neuron=first_neuron)


first_neuron=Neuron(dendrites=first_neuron_dendrite_list)

#Other neurons: second, third. Three dendrite and three axons

#Second Neuron
second_neuron_dendrite_1=Dendrite(initial_coord=(1,1))
second_neuron_dendrite_2=Dendrite(initial_coord=(2,2))
second_neuron_dendrite_3=Dendrite(initial_coord=(2,3))

second_neuron_dendrite_list=[second_neuron_dendrite_1, second_neuron_dendrite_2, second_neuron_dendrite_3]
second_neuron=Neuron(dendrites=second_neuron_dendrite_list)

second_neuron_axon_1=Axon((2,3), neuron=second_neuron)
second_neuron_axon_2=Axon((0,1), neuron=second_neuron)
second_neuron_axon_3=Axon((2,0), neuron=second_neuron)



#Third Neuron
third_neuron_dendrite_1=Dendrite(initial_coord=(2,2))
third_neuron_dendrite_2=Dendrite(initial_coord=(0,1))
third_neuron_dendrite_3=Dendrite(initial_coord=(2,0))

third_neuron_dendrite_list=[third_neuron_dendrite_1, third_neuron_dendrite_2, third_neuron_dendrite_3]
third_neuron=Neuron(dendrites=second_neuron_dendrite_list)

third_neuron_axon_1=Axon((1,3), neuron=third_neuron)
third_neuron_axon_2=Axon((1,0), neuron=third_neuron)
third_neuron_axon_3=Axon((2,1), neuron=third_neuron)

#More neurins with different dendritic functions: lower and minus

#Forth Neuron with Lower with five dendrites and three axons

forth_neuron_dendrite_1=Dendrite_Lower(initial_coord=(1,0))
forth_neuron_dendrite_2=Dendrite_Lower(initial_coord=(1,1))
forth_neuron_dendrite_3=Dendrite_Lower(initial_coord=(1,2))
forth_neuron_dendrite_4=Dendrite_Lower(initial_coord=(2,2))
forth_neuron_dendrite_5=Dendrite_Lower(initial_coord=(2,0))

forth_neuron_dendrite_list=[forth_neuron_dendrite_1,forth_neuron_dendrite_2,forth_neuron_dendrite_3,
                        forth_neuron_dendrite_4, forth_neuron_dendrite_5]
forth_neuron=Neuron(dendrites=forth_neuron_dendrite_list)

forth_neuron_axon_1=Axon((1,2), neuron=forth_neuron)
forth_neuron_axon_2=Axon((1,1), neuron=forth_neuron)
forth_neuron_axon_3=Axon((2,0), neuron=forth_neuron)

#Fifth Neuron with invert with five dendrite and three axons

fifth_neuron_dendrite_1=Dendrite_Minus(initial_coord=(2,0))
fifth_neuron_dendrite_2=Dendrite_Minus(initial_coord=(1,0))
fifth_neuron_dendrite_3=Dendrite_Minus(initial_coord=(1,2))
fifth_neuron_dendrite_4=Dendrite_Minus(initial_coord=(2,3))
fifth_neuron_dendrite_5=Dendrite_Minus(initial_coord=(1,1))

fifth_neuron_dendrite_list=[fifth_neuron_dendrite_1, fifth_neuron_dendrite_2, fifth_neuron_dendrite_3,
                            fifth_neuron_dendrite_4, fifth_neuron_dendrite_5]
fifth_neuron=Neuron(dendrites=fifth_neuron_dendrite_list)


fifth_neuron_axon_1=Axon((1,0), neuron=fifth_neuron)
fifth_neuron_axon_2=Axon((1,2), neuron=fifth_neuron)
fifth_neuron_axon_3=Axon((2,0), neuron=fifth_neuron)



list_of_axons=[first_neuron_axon_1, first_neuron_axon_2, first_neuron_axon_3,
               second_neuron_axon_1, second_neuron_axon_2, second_neuron_axon_3,
               third_neuron_axon_1, third_neuron_axon_2, third_neuron_axon_3,
               forth_neuron_axon_1, forth_neuron_axon_2, forth_neuron_axon_3,
               fifth_neuron_axon_1, fifth_neuron_axon_2, fifth_neuron_axon_3]

list_of_dendrites=[second_neuron_dendrite_1, second_neuron_dendrite_2, second_neuron_dendrite_3,
                   third_neuron_dendrite_1, third_neuron_dendrite_2, third_neuron_dendrite_3,
                   forth_neuron_dendrite_1, forth_neuron_dendrite_2, forth_neuron_dendrite_3,
                   forth_neuron_dendrite_4, forth_neuron_dendrite_5, fifth_neuron_dendrite_1,
                   fifth_neuron_dendrite_2, fifth_neuron_dendrite_3, fifth_neuron_dendrite_4,
                   fifth_neuron_dendrite_5]


list_of_neurons=[first_neuron, second_neuron, third_neuron, forth_neuron, fifth_neuron]

Node_1=Node(coordinates=(1,1))
Node_2=Node(coordinates=(1,2))
Node_3=Node(coordinates=(1,0))
Node_4=Node(coordinates=(2,2))
Node_5=Node(coordinates=(2,1))
Node_6=Node(coordinates=(2,3))
Node_7=Node(coordinates=(2,0))

list_of_nodes=[Node_1, Node_2, Node_3, Node_4, Node_5, Node_6, Node_7]

#Data

x=3
y=10

''' 

We even can draw a line through parabola. Linear regression style. But need to change sparkling_check function
and filtering in neuron. 

'''

# x=np.array((0,1,2,3,4)) #Not convinient with neuronal filter for inference: 
# # pass all data, but question with sparkling detection because of zeroes, maybe initialize obtained data with Nan?
# y=x**2-4+20+x**3



#Initialisation of the first dendrite receiving information

first_neuron_dendrite.obtained_signal=x
first_neuron_dendrite.connection_established=True



first_neuron_axon_1.axon_sparkling=True
first_neuron_axon_2.axon_sparkling=True
first_neuron_axon_3.axon_sparkling=True

#Functions for checking signals in different part of the net.
#Only for information

def check_signal_of_dendrite(list_of_dendrites):
    
    for dendrite in list_of_dendrites:
        
        print(f'signals: {dendrite.obtained_signal}')
        
        print(f'proceeding in dendrite: {dendrite.dendrite_proceed()}')

def neuron_logits_check(list_of_neurons):
    
    for neuron in list_of_neurons:
        
        neuron_logits=neuron.neuron_proceed()
        
        print(f'information in neuron: {neuron_logits}')


'''

Now it works.

Still there is a need of time-dependent features, some local dynamics in computations and attraction:
in dendrites and node energy counting.

Many experiments are waiting

Dendrites and axons behaviour, if there are no available points for connections
'''

#Main loop for training

for timestamp in range(Timestamps):
    
    first_neuron.child_dendrites_list=first_neuron_dendrite_list #renew by hand every time
    second_neuron.child_dendrites_list=second_neuron_dendrite_list #renew by hand every time
    third_neuron.child_dendrites_list=third_neuron_dendrite_list #renew by hand every time
    
    #Check signals in dendrites and neurons
    check_signal_of_dendrite(list_of_dendrites=list_of_dendrites)
    neuron_logits_check(list_of_neurons=list_of_neurons)
    
    
    #Sparkling neuron count
    count_sparkling_axons(list_of_nodes=list_of_nodes,list_of_axons=list_of_axons)
    
    #Energy in every node
    for node in list_of_nodes:
        node.energy_count()
    
    #Active dendrite count
    count_active_dendrites(list_of_dendrites=list_of_dendrites)
    print(f'list of active dendrites: {list_of_active_dendrites}')
    
    #All staff with moving and trying connections of dendrites
    for dendrite in list_of_active_dendrites:
        dendrite.moving()
        dendrite.try_connection()
        
        #This part is important. Only way to re-wright internal utility function in neuron
        if dendrite.UF_local>UF_max:
            UF_max=dendrite.UF_local
        UFglobal.append(UF_max)
        
        #Some information of the resting dendrites
        print(f'list of active dendrites: {len(list_of_active_dendrites)}')
        print(f'Obtained maximum utility function: {UF_max}')
    
    #Remove established dendrites i this run from list of active dendrites
    remove_from_active_dendrites()
    print(f'Obtained maximum utility function: {UF_max}')
    print(f'list of active dendrites: {len(list_of_active_dendrites)}')
