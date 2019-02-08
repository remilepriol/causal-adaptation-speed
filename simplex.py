import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np


diracs = np.eye(3)
uniform = np.ones(3)/3
centered_diracs = diracs - uniform
corners2d = np.array([[0, 2/np.sqrt(6)],
                      [-np.sqrt(2)/2, -1/np.sqrt(6)],
                      [np.sqrt(2)/2, -1/np.sqrt(6)],])
corners2d /= np.sqrt(2)


class Simplex:
    """Draw any function on a 2d simplex"""
    
    def __init__(self,corners=corners2d, resolution=6):
        self.corners = corners
        # change of basis matrices
        self.d3tod2 = np.linalg.solve(centered_diracs, corners) 
        # 3*3 and 3*2 ouputs 3*2
        self.d2tod3 = np.linalg.solve(corners[:2], centered_diracs[:2]) 
        # 2*2 and 2*3 outputs 2*3
        
        # triangular mesh
        self.triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
        self.refiner = tri.UniformTriRefiner(self.triangle)
        self.trimesh = self.refiner.refine_triangulation(subdiv=resolution)
        
        # coordinates
        self.cartesians = np.array([self.trimesh.x, self.trimesh.y]).T
        self.barycentrics = self.cart2bary(self.cartesians)
        print(f"Simplex with resolution {resolution}: {self.cartesians.shape[0]} points.")
                 
       
    def bary2cart(self,barys):
        """Take an N*3 array of barycenter coordinates
        and return an N*2 array of 2D cartesian coordinates."""
        return np.dot(barys - uniform, self.d3tod2)

    def cart2bary(self,cart):
        """Take an N*2 array of 2D cartesian coordinates
        and return an N*3 array of barycenter coordinates."""
        return np.dot(cart, self.d2tod3) + uniform
    
    def plotoptions(self):        
        plt.axis('equal')
        plt.axis('off')
        plt.tight_layout()
        
    
    def show_borders(self, color='blue'):
        # plot contours
        self.plotoptions()
        plt.triplot(self.triangle,color=color)
        
        # plot lines between center of triangle and midline of each edge
        center = np.mean(self.corners,axis=0)
        middles = 0.5*(self.corners + np.roll(self.corners,1,axis=0))
        for m in middles:
            plt.plot([m[0], center[0]],[m[1],center[1]], color=color)
        
        
    def show_mesh(self):
        self.plotoptions()
        plt.triplot(self.trimesh,linewidth=0.5)
    
    def show_func(self, baryfunc):
        values = baryfunc(self.barycentrics)
        self.show_borders()
        plt.tripcolor(self.trimesh, values)
        plt.colorbar()
        
    def constraint_line(self,eps,color='r'):
        """Plot the line of equation $q_1-q_2=eps$"""
        extremes = np.array([[(1+eps)/2, (1-eps)/2, 0],
                             [eps,0,1-eps]])
        cart = self.bary2cart(extremes)
        plt.plot(cart[:,0],cart[:,1],color=color)
        
    def scatter(self, barypoints, color='red'):
        carts = self.bary2cart(barypoints)
        if len(carts.shape)==1:
            carts = carts[np.newaxis,:]
        plt.scatter(carts[:,0],carts[:,1],
                    s=50, color=color,marker='*',zorder=3)
        
    def show_func3d(self, baryfunc):
        values = baryfunc(self.barycentrics)
        
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(5,5))
        ax = fig.gca(projection='3d')
        ax.triplot(self.triangle)
        # Plot the surface.
        surf = ax.plot_trisurf(self.cartesians[:,0], self.cartesians[:,1], values, 
                               cmap='viridis', linewidth=3, antialiased=False)
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()