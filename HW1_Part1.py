import numpy as np
import matplotlib.pyplot as plt

class Group():
    
    def __init__(self, identity, operation, inverse_func = None): 
       self.identity = identity
       self.operation = operation  
       self.inverse_func = inverse_func
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
        val = self.group.operation(self.val, ele.val)
        ele_new = GroupElement(val, self.group ) 
        return ele_new 

    def right_action(self, ele):
        val = self.group.operation( ele.val, self.val)
        ele_new = GroupElement(val, self.group) 
        return ele_new 
    
    def inverseElement(self):
        inv_val = self.group.inverse_func(self.val)
        ele_inv = GroupElement(inv_val, self.group)
        return ele_inv


def main(args=None):
   G = Group(1, composition, inverse_func) 
   g = G.element(np.array([0,1,-np.pi/4 ])) 
   h = G.element(np.array([1,2,-np.pi/2 ]))
   gh = g.left_action(h)
   hg = h.left_action(g)
   inverse_h = h.inverseElement()
   print("inverse h", inverse_h.val)
   inverse_g = g.inverseElement()
   print("inverse g", inverse_g.val)
   h_position_rel_g = inverse_g.left_action(h)
   g_position_rel_h = inverse_h.left_action(g) 
   print("h_rel_g", h_position_rel_g.val)
   print("_g, rel, h", g_position_rel_h.val) 


   plt.scatter(g.val[0], g.val[1], marker='x', label = 'G')
   plt.scatter(h.val[0], h.val[1], marker='x', label = 'H')
   plt.scatter(gh.val[0], gh.val[1], marker='x', label = 'GH')
   plt.scatter(hg.val[0], hg.val[1], marker='x', label = 'HG')
   plt.quiver(g.val[0],g.val[1],np.cos(g.val[2]), np.sin(g.val[2]))
   plt.quiver(h.val[0],h.val[1],np.cos(h.val[2]), np.sin(h.val[2]))
   plt.quiver(gh.val[0],gh.val[1],np.cos(gh.val[2]), np.sin(gh.val[2]))
   plt.quiver(hg.val[0],hg.val[1],np.cos(hg.val[2]), np.sin(hg.val[2]))
   plt.scatter(h_position_rel_g.val[0], h_position_rel_g.val[1], label = "g relative to h") 
   plt.scatter(g_position_rel_h.val[0],g_position_rel_h.val[1], label = "h relative to g")
   plt.quiver(h_position_rel_g.val[0], h_position_rel_g.val[1], np.cos(h_position_rel_g.val[2]),np.sin(h_position_rel_g.val[2]) )
   plt.quiver(g_position_rel_h.val[0], g_position_rel_h.val[1], np.cos(g_position_rel_h.val[2]),np.sin(g_position_rel_h.val[2]) )
   
   plt.title ("Part 1: All Deliverables Shown")
   plt.xlim([-3,3])
   plt.ylim([-3,3])
   plt.xlabel("x")
   plt.ylabel("y")
   plt.legend()
   plt.show()




def affine_addition(g1, g2):
    return (g1 +g2) 

def scalar_multiplication(g1,g2):
    return (g1 * g2)

def modular_addition(g1, g2, phi):
    return((g1+g2)% phi)

def composition(g1, g2,):
    el_1 = g2[0]*np.cos(g1[2])-g2[1]*np.sin(g1[2])+g1[0]
    el_2 = g2[0]*np.sin(g1[2])+g2[1]*np.cos(g1[2]) + g1[1]
    el_3 = g1[2]+g2[2]
    return [el_1, el_2, el_3]

def inverse_func(val):
    mat = np.array([[np.cos(val[2]), -1*np.sin(val[2]), val[0]], [np.sin(val[2]), np.cos(val[2]), val[1]], [0, 0, 1]])
    mat = np.linalg.inv(mat)
    theta = np.arctan2(mat[1][0], mat[1][1])
    x = mat[0][2]
    y = mat[1][2]
    return (np.array([x, y, theta]))

if __name__ == '__main__':
    main()