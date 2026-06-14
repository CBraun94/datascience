from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import _const as _c
from dataclasses import dataclass


@dataclass
class DataScatter3D:
    x: list = None
    y: list = None
    z: list = None


def plot_scatter3d(x, y, z, color, filename: str, fig: Figure = None, ax: Axes = None):
    ax.scatter(x, y, z, c=color, cmap='Accent', alpha=_c.ALPHA)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

