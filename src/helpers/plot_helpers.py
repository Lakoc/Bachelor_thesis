import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Wedge, Circle
from wordcloud import WordCloud
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import re


def gauge(labels, colors, min_val, max_val, value, tickers_format, val_format):
    """Create gauge graph for showing difference between overall and current data"""
    fig, ax = plt.subplots()

    if min_val < 0:
        max_val += min_val
        if max_val < 0:
            raise ValueError(f'Gauge intervals are not in range: max_val < 0({max_val})')
        min_val = max(0, min_val)

    # Set sector bounding
    mid_points = [20, 90, 160]
    ang_range = np.array([[0, 40], [40, 140], [140, 180]])

    labels = labels[::-1]

    # Create sectors
    patches = []
    for ang, c in zip(ang_range, colors):
        patches.append(Wedge((0., 0.), .4, *ang, width=0.10, facecolor=c, lw=2, alpha=0.5))

    [ax.add_patch(p) for p in patches]

    # Add texts for sectors
    for mid, lab in zip(mid_points, labels):
        ax.text(0.35 * np.cos(np.radians(mid)), 0.35 * np.sin(np.radians(mid)), lab, horizontalalignment='center',
                verticalalignment='center', fontsize=16,
                rotation=np.degrees(np.radians(mid) * np.pi / np.pi - np.radians(90)))

    # Add tickers
    tickers_range = np.linspace(0, 180, 10, endpoint=True)
    tickers_val = np.flip(np.linspace(min_val, max_val, 10, endpoint=True))
    tickers_val_arr = [tickers_format(val) for val in tickers_val]
    tickers_val_arr[0] = f'< {tickers_val_arr[0]}'
    tickers_val_arr[-1] = f'> {tickers_val_arr[-1]}'
    for ticker_angle, val in zip(tickers_range, tickers_val_arr):
        ax.text(0.420 * np.cos(np.radians(ticker_angle)), 0.420 * np.sin(np.radians(ticker_angle)), val,
                horizontalalignment='center',
                verticalalignment='center', fontsize=12,
                rotation=np.degrees(np.radians(ticker_angle) * np.pi / np.pi - np.radians(90)))
        ax.plot([0.395 * np.cos(np.radians(ticker_angle)), 0.405 * np.cos(np.radians(ticker_angle))],
                [0.395 * np.sin(np.radians(ticker_angle)), 0.405 * np.sin(np.radians(ticker_angle))], color='black',
                linewidth=.4)

    # Add value
    ax.text(-.014 * (len(val_format(value)) / 2), -0.08, val_format(value), fontsize=16)

    # Add arrow pointing on current value
    angle = value_to_angle(value, min_val, max_val)

    ax.arrow(0, 0, 0.225 * np.cos(np.radians(angle)), 0.225 * np.sin(np.radians(angle)), width=0.025, head_width=0.075,
             head_length=0.1, fc='k', ec='k')
    ax.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
    ax.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))

    # Disable axis
    ax.set_frame_on(False)
    ax.axes.set_xticks([])
    ax.axes.set_yticks([])
    ax.axis('equal')

    return fig, ax


def value_to_angle(value, min_val, max_val):
    """Return angle for gauge plot"""
    angle = (value - min_val) / (max_val - min_val)
    angle *= 180
    angle = max(min(angle, 180), 0)
    return 180 - angle


def word_cloud(text, path):
    """Generate a word cloud image"""
    colors = [to_rgb('C1'), to_rgb('C2'), to_rgb('C0')]
    my_cmap = LinearSegmentedColormap.from_list('cmap_default', colors, N=100)
    max_font_size = 140

    def color_map(_, **kwargs):
        color = to_rgb(my_cmap((kwargs['font_size'] / max_font_size) ** (1 / 2.2)))
        color_rgb_int = tuple(int(i * 255) for i in color)
        return color_rgb_int

    text = re.sub(r'[%_][a-zA-Z]*|{[a-zA-Z ]*}|\[[a-zA-Z ]*]|<[a-zA-Z]*>|[^a-zA-ZěščřžýáíéóúůďťňĎŇŤŠČŘŽÝÁĚÍÉÚŮ\s]', "", text).strip()
    out = WordCloud(background_color="white", max_font_size=max_font_size, color_func=color_map, width=800,
                    height=400).generate(text)

    fig, ax = plt.subplots(figsize=(16,9))
    ax.imshow(out, interpolation='bilinear')
    ax.axis("off")
    generate_graph(path, fig)
    return list(out.words_.keys())[0]


def generate_graph(path, fig):
    """Handler for graph generations"""
    fig.tight_layout()
    plt.savefig(path, transparent=True)
    plt.close(fig)
