from ground_truth.geomotion import representationliegroup as rlgp, plottingfunctions as gplt
from ground_truth.geomotion import group as gp
from ground_truth.geomotion import utilityfunctions as ut
import numpy as np
from matplotlib import pyplot as plt
from operator import methodcaller
import scipy as sc

spot_color = gplt.crimson



def SE2_rep(g_value):
    x = g_value[0]
    y = g_value[1]
    theta = g_value[2]

    g_rep = [[np.cos(theta), -np.sin(theta), x],
             [np.sin(theta), np.cos(theta), y],
             [0, 0, 1]]

    return g_rep


def SE2_derep(g_rep):
    x = g_rep[0][2]
    y = g_rep[1][2]
    theta = np.arctan2(g_rep[1][0], g_rep[0][0])

    g_value = [x, y, theta]
    return g_value


def SE2_normalize(g_rep):
    R = g_rep[0:2, 0:2]

    R_normalized = (1.5 * R) - (0.5 * np.matmul(np.matmul(R, np.transpose(R)), R))

    g_rep_normalized = np.concatenate([np.concatenate([R_normalized, g_rep[[0, 1], 2:]], 1), [[0, 0, 1]]])

    return (g_rep_normalized)


SE2 = rlgp.RepresentationLieGroup(SE2_rep, [0, 0, 0], SE2_derep, 0, SE2_normalize)

def SE3_rep(g_value):
    x = g_value[0]
    y = g_value[1]
    z = g_value[2]
    rot_x = g_value[3]
    rot_y = g_value[4]
    rot_z = g_value[5]

    r_matrix = sc.linalg.expm ([[0, -rot_z, rot_y], [rot_z, 0 , -rot_x], [-rot_y, rot_x, 0]])

    g_rep =np.array([[0, 0, 0, x], [0, 0, 0, y], [0, 0, 0, z], [0, 0, 0, 1]])
    g_rep[0:3,0:3] = r_matrix
    

    return g_rep 

def SE3_derep(g_rep):
    r_matrix = g_rep[0:3,0:3]
    r_matrix_derep = sc.linalg.logm(r_matrix)

    dereped_values = [g_rep[0,3], g_rep[1,3], g_rep[2,3], r_matrix_derep[2,1], r_matrix_derep[0,2], r_matrix_derep[1,0]]
    
    return dereped_values

SE3 = rlgp.RepresentationLieGroup(SE3_rep, [0, 0, 0, 0, 0, 0], SE3_derep, 0, None)

class RigidBodyPlotInfo:

    def __init__(self, **kwargs):

        if 'plot_locus' in kwargs:
            self.plot_locus = kwargs['plot_locus']
        else:
            self.plot_locus = None

        if 'plot_style' in kwargs:
            self.plot_style = kwargs['plot_style']
        else:
            self.plot_style = None

        if 'plot_function' in kwargs:
            self.plot_function = kwargs['plot_function']
        else:
            self.plot_function = ['fill']

        if 'plot_geometry' in kwargs:
            self.plot_geometry = kwargs['plot_geometry']
        else:
            self.plot_geometry = None


def cornered_triangle(configuration, r, spot_color, **kwargs):

    def T1(body):
        return SE2.element_set(ut.GridArray([[r, 0, 0],
                                       [r * np.cos(2 * np.pi / 3), r * np.sin(2 * np.pi / 3), 0],
                                       [r * np.cos(4 * np.pi / 3), r * np.sin(4 * np.pi / 3), 0]], 1),
                         0, "element")

    def T2(body):
        return SE2.element_set(ut.GridArray([[r, 0, 0],
                                       [r / 3 * np.cos(2 * np.pi / 3) + (2 * r / 3), r / 3 * np.sin(2 * np.pi / 3), 0],
                                       [r / 3 * np.cos(4 * np.pi / 3) + (2 * r / 3), r / 3 * np.sin(4 * np.pi / 3), 0]],
                                      1),
                         0, "element")

    plot_locus = [T1, T2]

    plot_style = [{"edgecolor": 'black', "facecolor": 'white'} | kwargs,
                  {"edgecolor": 'black', "facecolor": spot_color} | kwargs]

    plot_info = RigidBodyPlotInfo(plot_locus=plot_locus, plot_style=plot_style)

    return RigidBody(plot_info, configuration)


class RigidBody:

    def __init__(self,
                 plot_info,
                 position=SE2.identity_element()):
        self.plot_info = plot_info
        self.position = position

    def draw(self,
             axis):
        plot_locus = self.plot_info.plot_locus
        plot_options = self.plot_info.plot_style
        plot_function = self.plot_info.plot_function

        for i, p in enumerate(plot_locus):
            # Transform the locally expressed positions of the drawing points by the position of the body
            plot_locus_global = self.position * p(self)
            plot_locus_global_grid = plot_locus_global.grid

            if plot_function[i] == 'fill':
                axis.fill(*plot_locus_global_grid[:2], **(plot_options[i]))
            elif plot_function[i] == 'plot':
                axis.plot(*plot_locus_global_grid[:2], **(plot_options[i]))
            elif plot_function[i] == 'scatter':
                axis.scatter(*plot_locus_global_grid[:2], **(plot_options[i]))
            else:
                raise Exception("Unknown plot_function specification")
            #print(plot_points_global_grid[0], "\n", plot_points_global_grid[1])

        return
