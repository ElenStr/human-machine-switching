from collections import defaultdict
import os
import numpy as np

from matplotlib import pyplot as pl
from PIL import Image
from matplotlib import collections as mc
from environments.env import GridWorld

from definitions import ROOT_DIR

IMG_DIR = os.path.join(ROOT_DIR, 'cell_types/')

HUMAN_COLOR = 'orange'
MACHINE_COLOR = 'deepskyblue'


class PlotPath:
    """
    Plot an agent's trajectory in a grid-based environment during evaluation
    """

    def __init__(self, grid_world: GridWorld, n_try: int, img_dir: str = IMG_DIR):
        """
        Parameters
        ----------
        grid_world : GridWorld
            A grid world that the agent interacts with
        n_try : int
            The number of repeats (i.e., the number of trajectories)
        img_dir : str
            The directory of different cell type images

        """
        self.width = grid_world.width
        self.height = grid_world.height
        self.n_try = n_try

        self.cell_types = grid_world.cell_types
        self.img_dir = img_dir

        # lines {(org_x, org_y, dst_x, dst_y, color): count}
        self.lines = defaultdict(int)

    def add_line(self, src_state: tuple, dst_state: tuple, color: str):
        """
        Add line to the plot

        Parameters
        ----------
        src_state: tuple
            The start point (state) of the line
        dst_state: tuple
            The end point (state) of the line
        color : str
            The color of the line
        """
        src_x, src_y = src_state
        dst_x, dst_y = dst_state

        self.lines[(src_x, src_y, dst_x, dst_y, color)] += 1

    def plot(self, file_path: str):
        """
        Plot the result

        Parameters
        ----------
        file_path : str
            Name of the output plot
        """
        lines = []
        colors = []
        widths = []
        x_ratio = 0.5
        y_ratio = (self.height//10 )* self.width / self.height
        for line_tuple, count in self.lines.items():
            
            line = [((line_tuple[0] + 0.5) * x_ratio, (line_tuple[1] + 0.5) * y_ratio),
                    ((line_tuple[2] + 0.5) * x_ratio, (line_tuple[3] + 0.5) * y_ratio)]
            lines.append(line)
            colors.append(line_tuple[4])
            widths.append(count)

        widths = np.divide(widths, self.n_try / 5)
        lc = mc.LineCollection(lines, colors=colors, linewidths=widths)

        fig, ax = pl.subplots()

        # add images
        for x in range(self.width):
            for y in range(self.height):
                cell_type = str(self.cell_types[x, y])
                
                img = Image.open(self.img_dir + cell_type + '.png')
                ax.imshow(img, extent=((x * x_ratio), ((x + 1) * x_ratio), (y * y_ratio), ((y + 1) * y_ratio)))
        
        # add grid lines
        ratio = 10
        x = np.linspace(0, self.width, self.width * ratio)
        for y in range(self.height):
            ax.plot(x * x_ratio, [y * y_ratio for i in range(self.width * ratio)], color='black', linewidth=0.8)

        y = np.linspace(0, self.height, self.height * ratio)
        for x in range(self.width):
            ax.plot([x * x_ratio for i in range(self.height * ratio)], y * y_ratio, color='black', linewidth=0.8)

        ax.add_collection(lc)
        pl.grid(True, linewidth=1, color='gray')
        ax.set_xticks([])
        ax.set_yticks([])

        # plot
        fig.savefig(file_path, format='png', dpi=200, bbox_inches='tight', pad_inches=0.01)
        pl.close()

        # clear `self.lines`
        self.lines = defaultdict(int)

