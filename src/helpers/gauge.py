from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Wedge, Rectangle


def degree_range():
    start = [0, 40, 140]
    end = [40, 140, 180]
    mid_points = [20, 90, 160]
    return np.c_[start, end], mid_points


def rot_text(ang):
    rotation = np.degrees(np.radians(ang) * np.pi / np.pi - np.radians(90))
    return rotation


def gauge(labels, colors='jet_r', angle=0, title='', fname=False):
    """
    some sanity checks first

    """

    labels_size = len(labels)

    """
    if colors is a string, we assume it's a matplotlib colormap
    and we discretize in N discrete colors 
    """

    if isinstance(colors, str):
        cmap = cm.get_cmap(colors, labels_size)
        cmap = cmap(np.arange(labels_size))
        colors = cmap[::-1, :].tolist()
    if isinstance(colors, list):
        if len(colors) == labels_size:
            colors = colors[::-1]
        else:
            raise Exception("\n\nnumber of colors {} not equal \
            to number of categories{}\n".format(len(colors), labels_size))

    """
    begins the plotting
    """

    fig, ax = plt.subplots()

    ang_range, mid_points = degree_range()

    labels = labels[::-1]

    """
    plots the sectors and the arcs
    """
    patches = []
    for ang, c in zip(ang_range, colors):
        # sectors
        patches.append(Wedge((0., 0.), .4, *ang, facecolor='w', lw=2))
        # arcs
        patches.append(Wedge((0., 0.), .4, *ang, width=0.10, facecolor=c, lw=2, alpha=0.5))

    [ax.add_patch(p) for p in patches]

    """
    set the labels (e.g. 'LOW','MEDIUM',...)
    """

    for mid, lab in zip(mid_points, labels):
        ax.text(0.35 * np.cos(np.radians(mid)), 0.35 * np.sin(np.radians(mid)), lab, horizontalalignment='center',
                verticalalignment='center', fontsize=14, fontweight='bold', rotation=rot_text(mid))

    """
    set the bottom banner and the title
    """
    r = Rectangle((-0.4, -0.1), 0.8, 0.1, facecolor='w', lw=2)
    ax.add_patch(r)

    ax.text(0, -0.05, title, horizontalalignment='center', verticalalignment='center', fontsize=22, fontweight='bold')

    """
    plots the arrow now
    """

    ax.arrow(0, 0, 0.225 * np.cos(np.radians(angle)), 0.225 * np.sin(np.radians(angle)), width=0.025, head_width=0.075,
             head_length=0.1, fc='k', ec='k')

    ax.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
    ax.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))

    """
    removes frame and ticks, and makes axis equal and tight
    """

    ax.set_frame_on(False)
    ax.axes.set_xticks([])
    ax.axes.set_yticks([])
    ax.axis('equal')
    plt.tight_layout()
    # fig.show()
    if fname:
        fig.savefig(fname)
