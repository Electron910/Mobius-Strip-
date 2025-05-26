import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import simps
from scipy.spatial.distance import euclidean

class MobiusStrip:
    def __init__(self, R=1.0, w=0.2, n=200):
        self.R = R
        self.w = w
        self.n = n
        self.u, self.v = np.meshgrid(
            np.linspace(0, 2 * np.pi, n),
            np.linspace(-w/2, w/2, n)
        )
        self.x, self.y, self.z = self.compute_coordinates()

    def compute_coordinates(self):
        u = self.u
        v = self.v
        x = (self.R + v * np.cos(u / 2)) * np.cos(u)
        y = (self.R + v * np.cos(u / 2)) * np.sin(u)
        z = v * np.sin(u / 2)
        return x, y, z

    def surface_area(self):
        dxdu = np.gradient(self.x, axis=1)
        dxdv = np.gradient(self.x, axis=0)
        dydu = np.gradient(self.y, axis=1)
        dydv = np.gradient(self.y, axis=0)
        dzdu = np.gradient(self.z, axis=1)
        dzdv = np.gradient(self.z, axis=0)

        cross_x = dydu * dzdv - dzdu * dydv
        cross_y = dzdu * dxdv - dxdu * dzdv
        cross_z = dxdu * dydv - dydu * dxdv

        area_element = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2)
        du = 2 * np.pi / self.n
        dv = self.w / self.n

        return np.sum(area_element) * du * dv

    def edge_length(self):
        
        edge1 = np.array([
            (self.x[i, -1], self.y[i, -1], self.z[i, -1]) for i in range(self.n)
        ])
        edge2 = np.array([
            (self.x[i, 0], self.y[i, 0], self.z[i, 0]) for i in reversed(range(self.n))
        ])
        full_edge = np.vstack((edge1, edge2))

        return sum(euclidean(full_edge[i], full_edge[i + 1]) for i in range(len(full_edge) - 1))

    def plot(self):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.x, self.y, self.z, cmap='viridis', edgecolor='none')
        ax.set_title("Mobius Strip")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    strip = MobiusStrip(R=1.0, w=0.4, n=300)
    area = strip.surface_area()
    length = strip.edge_length()

    print(f"Surface Area ≈ {area:.4f}")
    print(f"Edge Length ≈ {length:.4f}")

    strip.plot()
