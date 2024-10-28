import numdifftools as nd 
import math
import numpy as np
import matplotlib.pyplot as plt

class Group():
    
    def __init__(self, identity, inverse, representation, derepresentation): 
       self.identity = identity
       self.representation = representation
       self.derepresentation = derepresentation
       self.inverse_func = inverse

       #operation is a function that performs the operation

    def element(self, val):
        return GroupElement(val, self)
    
    def identity_element(self):
        return (self.element(self.identity))



class GroupElement():
    def __init__(self, val, group):
        self.group = group
        self.val = val


    def left_action(self, ele):
        g1_rep = self.group.representation(self.val)
        ele_rep = ele.group.representation(ele.val)
        new_mat = g1_rep @ ele_rep
        val = self.group.derepresentation(new_mat)
        ele_new = GroupElement(val, self.group ) 
        return ele_new 

    def left_lifted_action(self, h, h_dot):
        mat = self.group.representation(self.val) @ h.group.representation(h.val)
        val = self.group.derepresentation(mat)
        return GroupElement(val, self.group)
    
    def action_helper(self, mat):
        return np.array([[mat[0]], [0], [mat[1]],[1]])


    def right_lifted_action(self, h, h_dot):
        mat =  h.group.representation(h.val) @ self.group.representation(self.val)
        val = self.group.derepresentation(mat)
        return GroupElement(val, self.group)


    def right_action(self, ele):
        g1_rep = self.group.representation(self.val)
        ele_rep = ele.group.representation(ele.val)
        new_mat =  ele_rep @ g1_rep
        val = self.group.derepresentation(new_mat)
        ele_new = GroupElement(val, self.group ) 
        return ele_new 

    def inverseElement(self):
        mat = self.group.representation(self.val)
        inv_val = self.group.inverse_func(mat)
        inv_val = self.group.derepresentation(inv_val)
        ele_inv = GroupElement(inv_val, self.group)
        return ele_inv
    
    def AD(self, ele):
        rep = self.group.representation(self.val) @ ele.group.representation(ele.val) @ np.linalg.inv(self.group.representation(self.val))
        derep = self.group.derepresentation(rep)
        return GroupElement(derep, self.group)
    
    def AD_inverse (self, ele):
       rep = np.linalg.inv(self.group.representation(self.val)) @ ele.group.representation(ele.val) @ self.group.representation(self.val)
       derep = self.group.derepresentation(rep)
       return GroupElement(derep, self.group)
    
    

    
    

class Vector_Tangent():
    def  __init__(self, config, velocity, velocity_derep): 
        #input: group (in matrix form)

        self.config  = config # this is now a group element 
        self.velocity = velocity # what is the vector 
        self.velocity_derep = velocity_derep
     
    def ThLg (self, h, h_dot):
        #TO DO: HOW DOES THIS WORK 
        JL = self.config.right_lifted_action(h, h_dot) 
        print("JL: ", JL.val )
        mat = self.velocity_derep(h_dot.group.representation(h_dot.val))
        v =  np.array(mat) @ np.array(JL.group.representation(JL.val))
        return Vector_Tangent(self.config,v, self.velocity_derep )

    def ThRg (self, h, h_dot):
        #HOW DOES THIS 
        JL = self.config.left_lifted_action(h ,h_dot)
        print("JL: ", JL.val )
        mat = self.velocity_derep(h_dot.group.representation(h_dot.val))
        v =   np.array(JL.group.representation(JL.val)) @ np.array(mat)
        return Vector_Tangent(self.config,v, self.velocity_derep)
    
    def Ad (self,g, g_dot):
        g_inverse= g.inverseElement()
        Ad = g.group.representation(g.val) @ g_dot.group.representation(g_dot.val) @ g_inverse.group.representation(g_inverse.val)
        velocity = self.config.group.representation(self.config.val) @ Ad
        velocity_derep = self.velocity_derep(velocity)
        return Vector_Tangent(self.config,velocity_derep, self.velocity_derep)

    def Ad_inv (self, g, g_dot):
        g_inverse= g.inverseElement()
        Ad = g_inverse.group.representation(g_inverse.val) @ g_dot.group.representation(g_dot.val) @ g.group.representation(g.val)
        velocity = self.config.group.representation(self.config.val) @ Ad
        velocity_derep = self.velocity_derep(velocity)
        return Vector_Tangent(self.config,velocity_derep, self.velocity_derep)

    
    

def plot(configs, title ):
    fig,ax = plt.subplots()
    for config in configs: 
        plt.quiver(config.config[0], config.config[1],config.velocity[0], config.velocity[1], angles = 'xy', scale_units = 'xy', scale = 3)
    ax.set_aspect ("equal")
    plt.title(title)
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.show()

def plot2(configs_x, configs_y, title):
    #print(configs_x )
    fig,ax = plt.subplots()
    for config_x in configs_x: 
        #print("config_x,config: ", config_x.config)
        plt.quiver(config_x.config.val[0], config_x.config.val[1],config_x.velocity[0], config_x.velocity[1], angles = 'xy', scale_units = 'xy', scale = 4, color = 'k')
    
    for config_y in configs_y: 
        plt.quiver(config_y.config.val[0], config_y.config.val[1],config_y.velocity[0], config_y.velocity[1], angles = 'xy', scale_units = 'xy', scale = 4, color = 'r')
    ax.set_aspect ("equal")
    plt.title(title)
    plt.xlim([0, 3])
    plt.ylim([-2, 2])
    plt.show()

def plot3(configs_x, configs_y, title):
    
    fig,ax = plt.subplots()
    for config_x in configs_x: 
        #print("config_x,config: ", config_x.config)
        plt.quiver(config_x.config.val[0], config_x.config.val[1],config_x.velocity[0], config_x.velocity[1], angles = 'xy', scale_units = 'xy', scale = 4, color = 'k')
    
    for config_y in configs_y: 
        plt.quiver(config_y.config.val[0], config_y.config.val[1],config_y.velocity[0], config_y.velocity[1], angles = 'xy', scale_units = 'xy', scale = 4, color = 'r')
    ax.set_aspect ("equal")
    plt.title(title)
    plt.xlim([0, 3])
    plt.ylim([-2, 2])
    plt.show()



def main():
    entry_points = []
    for x in range (0,5):
        for y in range (-2,3):
            if x != 0: 
                entry_points.append ([x/2,y/2])

    #Part 6
    
    
    G = Group(1,inverse_func, representation, derepresentation) 
    h1 = GroupElement([1,0], G)
    h2 = GroupElement([1,0], G)
    h_dot1 =GroupElement([1,0], G)
    h_dot2 = GroupElement([0,1], G)
    configs_x_lifted = []
    configs_y_lifted = []

    for pt in entry_points: 
        entry_point = GroupElement(pt, G)
        config = Vector_Tangent(entry_point, 0, velocity_derep)
        config_x = config.ThLg(h1, h_dot1)
        configs_x_lifted.append(config_x)
        config_y = config.ThLg(h2, h_dot2)
        configs_y_lifted.append(config_y)

    print("Config x: ", config_x.config.val)
    print(config_x.velocity)
    plot2(configs_x_lifted, configs_y_lifted, "Left Lifted Action")
    
    print("RIGHT")
   
    configs_x_lifted_right = []
    configs_y_lifted_right = []

    for pt in entry_points: 
        entry_point = GroupElement(pt, G)
        config = Vector_Tangent(entry_point, 0, velocity_derep)
        config_x = config.ThRg(h1, h_dot1)
        configs_x_lifted_right.append(config_x)
        config_y = config.ThRg(h2, h_dot2)
        configs_y_lifted_right.append(config_y)

    plot2(configs_x_lifted_right, configs_y_lifted_right, "Right Lifted Action")
    
    # Part 6 Adjoint
    
    G = Group(1,inverse_func, representation, derepresentation) 
    g = GroupElement([0.5,-1], G)
 
    g_circ = GroupElement([1,0.5], G)
    
    configs_adjoint = []
    configs_adjoint_inv = []
    

    for pt in entry_points: 
        entry_point = GroupElement(pt, G)
        config = Vector_Tangent(entry_point, 0, velocity_derep)
        configs = config.Ad(g, g_circ)
        configs_adjoint.append(configs)
        configs_inv = config.Ad_inv(g, g_circ)
        configs_adjoint_inv.append(configs_inv)
       
    print("configs: ", configs.velocity)
    plot3(configs_adjoint,configs_adjoint_inv, "Adjoint and Adjoint Inverse with G_circ with matrix representation")
    '''
    vel = [[1,2], [3,4]]
    print(velocity_derep(vel))
    '''






'''
def representation(g1):
    return np.array([[np.cos(g1[2]), -1*np.sin(g1[2]), g1[0]], [np.sin(g1[2]), np.cos(g1[2]), g1[1]], [0, 0, 1]])
'''
def representation(v):
    return np.array([[v[0],v[1]], [0, 1]])
'''
def derepresentation(mat):
    theta = np.arctan2(mat[1][0], mat[1][1])
    x = mat[0][2]
    y = mat[1][2]
    return (np.array([x, y, theta]))
'''
def derepresentation(mat):
    x = mat[0][0]
    y = mat[0][1]
    return ([x,y])

def inverse_func(mat):
    return np.linalg.inv(mat)

def velocity_derep(velocity):
    # used equation 2.175 in the book, not sure if that is correct 
    row, col =  np.array(velocity).shape
    vectorized = []
    for j in range (0, col):
        for i in range (0, row):
            vectorized.append(velocity[i][j]) 
            

    #vel_derep = np.linalg.pinv(np.array(velocity)) 
    
    d_dp = np.array([[1,0], [0, 0], [0,1], [0, 0]])
    #d_dp= d_dp.T
    d_dp = np.linalg.pinv(np.array(d_dp)) 
    #print("d_dp: ", d_dp)
    vectorized = np.array(vectorized)

    #print("vectorized: ", vectorized)

    vel_derep = d_dp@ vectorized
    

    return vel_derep


main()