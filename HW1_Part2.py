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
    


    

def main(args=None):
   G = Group(1,inverse_func, representation, derepresentation) 
   g = G.element(np.array([0,1,-np.pi/4 ])) 
   h = G.element(np.array([1,2,-np.pi/2 ]))


   gh = g.left_action(h)
   hg = h.left_action(g)
   inverse_g = g.inverseElement()
   inverse_h = h.inverseElement()



   h_position_rel_g = inverse_g.left_action(h)
   g_position_rel_h = inverse_h.left_action(g) 
   print("h rel g" , h_position_rel_g.val)

   
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
   
   plt.title ("Part 2: All Deliverables Shown")
   plt.xlim([-3,3])
   plt.ylim([-3,3])
   plt.xlabel("x")
   plt.ylabel("y")
   plt.legend()
   plt.show()

   ### Part 3 
   G = Group(1,inverse_func, representation, derepresentation) 
   g = G.element(np.array([0,1,-np.pi/4 ])) 
   h = G.element(np.array([1,2,-np.pi/2 ]))

   inverse_g = g.inverseElement()
   inverse_h = h.inverseElement()
   h_position_rel_g = inverse_g.left_action(h)
   AD_of_rel_pos = g.AD(h_position_rel_g)
   new = AD_of_rel_pos.left_action(g)
   print("G2 val: ", h.val)
   print("AD of g1 val: ", new.val)

   plt.scatter(g.val[0], g.val[1], marker='x', label = 'G')
   plt.scatter(AD_of_rel_pos.val[0], AD_of_rel_pos.val[1], marker='x', label = 'AD_of_rel_pos')
   plt.scatter(new.val[0], new.val[1], marker='x', label = 'G1 with AD_of_rel_pos')
  
   plt.quiver(g.val[0],g.val[1],np.cos(g.val[2]), np.sin(g.val[2]))
   plt.quiver(AD_of_rel_pos.val[0],AD_of_rel_pos.val[1],np.cos(AD_of_rel_pos.val[2]), np.sin(AD_of_rel_pos.val[2]))
   plt.quiver(new.val[0],new.val[1],np.cos(new.val[2]), np.sin(new.val[2]))
   
   
   plt.title ("Part 3: Deliverable 1")
   plt.xlim([-3,3])
   plt.ylim([-3,3])
   plt.xlabel("x")
   plt.ylabel("y")
   plt.legend()
   plt.show()

    #deleriable 2

   ## assuming that H2 is G2 in other system 
   h1 = G.element(np.array([-1,0,np.pi/2 ]))
   g2 = G.element(np.array([1,2,-np.pi/2 ]))
   gh1 = g.left_action(h1)
   g1_inv = g.inverseElement()
   h21 = g1_inv.left_action(g2)
   AD_inv = g.AD_inverse(h1)
   h2 = AD_inv.mult_two_pos(h1) 
   print("H2: ", h2.val)
   gh2 = g2.left_action(h2)
   AD_inv_of_21 = h21.AD_inverse(h1)
   print("here", h2.val)





   plt.scatter(g.val[0], g.val[1], marker='x', label = 'G')
   plt.scatter(g2.val[0], g2.val[1], marker='x', label = 'G2')
   plt.scatter(h1.val[0], h1.val[1], marker='x', label = 'H1')
   plt.scatter(h2.val[0], h2.val[1], marker='x', label = 'H2')
   plt.scatter(h21.val[0], h21.val[1], marker='x', label = 'H21')
   plt.scatter(gh1.val[0], gh1.val[1], marker='x', label = 'GH1')
   plt.scatter(gh2.val[0], gh2.val[1], marker='x', label = 'GH2')
   plt.scatter(AD_inv_of_21.val[0], AD_inv_of_21.val[1], marker='x', label = 'AD inverse of h21 ')

  
   plt.quiver(g.val[0],g.val[1],np.cos(g.val[2]), np.sin(g.val[2]))
   plt.quiver(g2.val[0],g2.val[1],np.cos(g2.val[2]), np.sin(g2.val[2]))
   plt.quiver(h1.val[0],h1.val[1],np.cos(h1.val[2]), np.sin(h1.val[2]))
   plt.quiver(h2.val[0],h2.val[1],np.cos(h2.val[2]), np.sin(h2.val[2]))
   plt.quiver(h21.val[0],h21.val[1],np.cos(h21.val[2]), np.sin(h21.val[2]))
   plt.quiver(gh1.val[0],gh1.val[1],np.cos(gh1.val[2]), np.sin(gh1.val[2]))
   plt.quiver(gh2.val[0],gh2.val[1],np.cos(gh2.val[2]), np.sin(gh2.val[2]))
   plt.quiver(AD_inv_of_21.val[0],AD_inv_of_21.val[1],np.cos(AD_inv_of_21.val[2]), np.sin(AD_inv_of_21.val[2]))
  
   
   
   plt.title ("Part 3: Deliverable 2")
   plt.xlim([-4,4])
   plt.ylim([-4,4])
   plt.xlabel("x")
   plt.ylabel("y")
   plt.legend()
   plt.show()

   











def representation(g1):
    return np.array([[np.cos(g1[2]), -1*np.sin(g1[2]), g1[0]], [np.sin(g1[2]), np.cos(g1[2]), g1[1]], [0, 0, 1]])

def derepresentation(mat):
    theta = np.arctan2(mat[1][0], mat[1][1])
    x = mat[0][2]
    y = mat[1][2]
    return (np.array([x, y, theta]))


def inverse_func(mat):
    return np.linalg.inv(mat)

if __name__ == '__main__':
    main()