import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Wedge, Circle
from wordcloud import WordCloud


def gauge(labels, colors, angle, path):
    """Create gauge graph for showing difference between overall and current data"""
    fig, ax = plt.subplots()

    # Set sector bounding
    mid_points = [20, 90, 160]
    ang_range = np.array([[0, 40], [40, 140], [140, 180]])

    labels = labels[::-1]

    # Create sectors
    patches = []
    for ang, c in zip(ang_range, colors):
        patches.append(Wedge((0., 0.), .4, *ang, facecolor='w', lw=2))
        patches.append(Wedge((0., 0.), .4, *ang, width=0.10, facecolor=c, lw=2, alpha=0.5))

    [ax.add_patch(p) for p in patches]

    # Add texts for sectors
    for mid, lab in zip(mid_points, labels):
        ax.text(0.35 * np.cos(np.radians(mid)), 0.35 * np.sin(np.radians(mid)), lab, horizontalalignment='center',
                verticalalignment='center', fontsize=14, fontweight='bold',
                rotation=np.degrees(np.radians(mid) * np.pi / np.pi - np.radians(90)))

    # Add arrow pointing on current value
    ax.arrow(0, 0, 0.225 * np.cos(np.radians(angle)), 0.225 * np.sin(np.radians(angle)), width=0.025, head_width=0.075,
             head_length=0.1, fc='k', ec='k')
    ax.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
    ax.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))

    # Disable axis
    ax.set_frame_on(False)
    ax.axes.set_xticks([])
    ax.axes.set_yticks([])
    ax.axis('equal')

    # Crate graph
    generate_graph(path, fig)


def value_to_angle(value, center, variance):
    """Return angle for gauge plot"""
    angle = (center - value) / variance * 5 / 9
    angle = min(max(angle, -1), 1)
    angle += 1
    angle *= 90
    return angle


def word_cloud(text, path):
    """Generate a word cloud image"""
    out = WordCloud(background_color="white").generate(text)

    fig, ax = plt.subplots()
    ax.imshow(out, interpolation='bilinear')
    ax.axis("off")
    generate_graph(path, fig)


def generate_graph(path, fig):
    """Handler for graph generations"""
    fig.tight_layout()
    plt.savefig(path)
    plt.close(fig)
