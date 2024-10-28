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

    def left_lifted_action(self, ele):
        eleval = [ele]
        JLg = nd.Jacobian (lambda eleval: self.left_action_helper(self, eleval) )(self.val)
        return JLg


    def left_action_helper(self, val, hval):
            hval_ele = GroupElement(hval, self.group )
            return (val.group.representation(val.val) @ hval_ele.group.representation(hval_ele.val) ) 

    def right_lifted_action(self, ele):
        #eleval = ele.val
        eleval = [ele]
        JLg = nd.Jacobian (lambda eleval: self.right_action_helper(self, eleval) )(self.val) 
        return JLg


    def right_action_helper(self, val, hval):
            #print("val: ", val.group.representation(val.val))
            hval_ele = GroupElement(hval, self.group )
            #print("hval: ", hval_ele.group.representation(hval_ele.val))
            #print("value of mult: ",hval_ele.group.representation(hval_ele.val) @ val.group.representation(val.val))
            return (hval_ele.group.representation(hval_ele.val) @ val.group.representation(val.val) ) 

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
    
    def mult_two_pos (self, ele):
        rep_self =  self.group.representation(self.val)
        rep_ele = self.group.representation(ele.val)
        new = rep_self @ rep_ele
        derep = self.group.derepresentation(new)
        return GroupElement(derep, self.group)
    

class Vector_Tangent():
    def  __init__(self, config, velocity): 
        #input: group (in matrix form)

        self.config  = config # where is the vector located  
        self.velocity = velocity # what is the vector 
    
    def derivative_in_the_direction(self, func, eval_point):
        new_config = lambda delta:func(eval_point, delta)

        new_velocity = nd.Derivative(new_config)(0) 
      
        return Vector_Tangent(eval_point,new_velocity )

    def derivative_in_the_direction_jacobian(self, func, eval_point):

        new_velocity = nd.Jacobian(lambda delta:func(eval_point, delta))(0)
      
        return Vector_Tangent(eval_point, new_velocity )
    
    def direction_of_derivative_group_action(self, pt, group):
        dx = nd.Jacobian(lambda delta:self.helper_with_group_action(pt, delta, 0, group))(0)
        dy = nd.Jacobian(lambda delta:self.helper_with_group_action(pt, delta, 1, group))(0)
        x_config = Vector_Tangent(pt.val, dx ) 
        y_config = Vector_Tangent(pt.val, dy)
        return  (x_config, y_config)
        
    def helper_with_group_action (self,pt,delta, idx, group):
        vector = [0,0]
        vector[idx] = delta[0]
        g = GroupElement(vector, group)
        return np.array(g.left_action(pt).val)
    
    def direction_of_derivative_group_action_right(self, pt, group):
        dx = nd.Jacobian(lambda delta:self.helper_with_group_action_right(pt, delta, 0, group))(0)
        dy = nd.Jacobian(lambda delta:self.helper_with_group_action_right(pt, delta, 1, group))(0)
        x_config = Vector_Tangent(pt.val, dx ) 
        y_config = Vector_Tangent(pt.val, dy)
        return  (x_config, y_config)
        
    def helper_with_group_action_right (self,pt,delta, idx, group):
        vector = [0,0]
        vector[idx] = delta[0]
        g = GroupElement(vector, group)
        return np.array(g.right_action(pt).val)
    
    def ThLg (self, h, entry_point, h_dot):
        JL = entry_point.right_lifted_action(h) 
        #print("JL: ", JL)
        JL = np.array(JL[0])
        #print("JP NP Array: ", JL)
        v = h_dot @ JL

        return Vector_Tangent(entry_point.val,v )

    def ThRg (self, h, entry_point, h_dot):
        JL = entry_point.left_lifted_action(h)
        #print("JL: ", JL) 
        #print("JL[0]: ", JL[0]) 
        JL = np.array(JL[0])
        v =   JL @ h_dot 

        return Vector_Tangent(entry_point.val,v )

    def helper_with_group_action_lifted (self,pt,delta, idx, group):
        vector = [0,0]
        vector[idx] = delta[0]
        g = GroupElement(vector, group)
        return np.array(g.left_action_lifted(pt).val)
    
    def Ad (self,entry_point, g, g_dot):
        Ad = g.group.inverse_func(g.right_lifted_action(g.left_lifted_action(g_dot.val))[0])
        
        velocity = (entry_point.val) @ Ad
        
        return Vector_Tangent(self.config, velocity)

    def Ad_inv (self,entry_point, g, g_dot):
        Ad = g.group.inverse_func(g.left_lifted_action(g.right_lifted_action(g_dot.val))[0])
        
        velocity = (entry_point.val) @ Ad
        return Vector_Tangent(self.config, velocity)

    

def f1 (config, delta):
    p = 1 + delta/math.sqrt(config[0]**2+config[1]**2)
    theta =[[math.cos(delta), -1*math.sin(delta)],[math.sin(delta),math.cos(delta)]]
    config_matrix = np.array([config[0], config[1]])
    #return (p * (theta @ config_matrix.reshape(2, 1)))
    return (p* np.matmul(theta, config_matrix))

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
        plt.quiver(config_x.config[0], config_x.config[1],config_x.velocity[0], config_x.velocity[1], angles = 'xy', scale_units = 'xy', scale = 4, color = 'r')
    
    for config_y in configs_y: 
        plt.quiver(config_y.config[0], config_y.config[1],config_y.velocity[0], config_y.velocity[1], angles = 'xy', scale_units = 'xy', scale = 4, color = 'k')
    ax.set_aspect ("equal")
    plt.title(title)
    plt.xlim([0, 3])
    plt.ylim([-2, 2])
    plt.show()

def plot3(configs_x, configs_y, title):
    
    fig,ax = plt.subplots()
    for config_x in configs_x: 
        #print("config_x,config: ", config_x.config)
        plt.quiver(config_x.config[0], config_x.config[1],config_x.velocity[0], config_x.velocity[1], angles = 'xy', scale_units = 'xy', scale = 8, color = 'r')
    
    for config_y in configs_y: 
        plt.quiver(config_y.config[0], config_y.config[1],config_y.velocity[0], config_y.velocity[1], angles = 'xy', scale_units = 'xy', scale = 8, color = 'k')
    ax.set_aspect ("equal")
    plt.title(title)
    plt.xlim([0, 3])
    plt.ylim([-2, 2])
    plt.show()



def main():

    #part 2 
    
    configs = []
    entry_points = []

    for x in range (-2,3):
        for y in range (-2,3):
            entry_points.append ([x,y])
    #'''
    for pt in entry_points:
        config = Vector_Tangent(pt, 0) 
        new_config = config.derivative_in_the_direction(f1, pt)
        configs.append(new_config)

    plot(configs, "Direction of Derivative with Derivative")
    
    for pt in entry_points: 
        config_jacobian = Vector_Tangent(pt, 0)
        config_jacobian.derivative_in_the_direction_jacobian(f1, pt)
        configs.append(new_config)

    plot(configs, "Direction of Derivative with Jacobian")
    #'''
    entry_points = []
    for x in range (0,5):
        for y in range (-2,3):
            if x != 0: 
                entry_points.append ([x/2,y/2])

    
    #'''
    #part 3
    G = Group(1,inverse_func, representation, derepresentation) 
    configs_x = []
    configs_y = []
    for pt in entry_points: 
        entry_point = GroupElement(pt, G)

        config = Vector_Tangent(pt, 0)
        configs = config.direction_of_derivative_group_action(entry_point,G)
        configs_x.append (configs[0])
        configs_y.append (configs[1])

    plot2(configs_x, configs_y, "Over Partial Lg")

    configs_x_right = []
    configs_y_right = []
    for pt in entry_points: 
        entry_point = GroupElement(pt, G)

        config = Vector_Tangent( pt, 0)
        configs = config.direction_of_derivative_group_action_right(entry_point,G)
        configs_x_right.append (configs[0])
        configs_y_right.append (configs[1])

    plot2(configs_x_right, configs_y_right, "Over Partial Rh")
    #'''
    #Part 4
    
    G = Group(1,inverse_func, representation, derepresentation) 
    h1 = GroupElement([1,0], G)
    h2 = GroupElement([1,0], G)
    h_dot1 =np.array([1,0])
    h_dot2 = np.array([0,1])
    configs_x_lifted = []
    configs_y_lifted = []

    for pt in entry_points: 
        config = Vector_Tangent(pt, 0)
        entry_point = GroupElement(pt, G)
        config_x = config.ThLg(h1, entry_point, h_dot1)
        configs_x_lifted.append(config_x)
        config_y = config.ThLg(h2, entry_point, h_dot2)
        configs_y_lifted.append(config_y)

    plot2(configs_x_lifted, configs_y_lifted, "Left Lifted Action")
    

    G = Group(1,inverse_func, representation, derepresentation) 
    h1 = GroupElement([1,0], G)
    h2 = GroupElement([1,0], G)
    h_dot1 =np.array([1,0])
    h_dot2 = np.array([0,1])
    configs_x_lifted_right = []
    configs_y_lifted_right = []

    for pt in entry_points: 
        config = Vector_Tangent(pt, 0)
        entry_point = GroupElement(pt, G)
        config_x = config.ThRg(h1, entry_point, h_dot1)
        configs_x_lifted_right.append(config_x)
        config_y = config.ThRg(h2, entry_point, h_dot2)
        configs_y_lifted_right.append(config_y)

    plot2(configs_x_lifted_right, configs_y_lifted_right, "Right Lifted Action")

    # Part 4 Adjoint

    G = Group(1,inverse_func, representation, derepresentation) 
    g = GroupElement([0.5,-1], G)
    #g = GroupElement([1.5,1], G)
 
    g_circ = GroupElement([1,0.5], G)
    #g_circ_left = GroupElement([1,-0.25], G)
    
    configs_adjoint = []
    configs_adjoint_inv = []
    

    for pt in entry_points: 
        config = Vector_Tangent(pt, 0)
        entry_point = GroupElement(pt, G)
        configs = config.Ad(entry_point,g, g_circ)
        #Maybe comment out 
        
        g_circ_left = GroupElement(configs.velocity, G)
        configs_adjoint.append(configs)
        configs_inv = config.Ad_inv(entry_point,g, g_circ_left)
        configs_adjoint_inv.append(configs_inv)
       

    plot3(configs_adjoint,configs_adjoint_inv, "Adjoint and Adjoint Inverse with G_circ")



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




main()