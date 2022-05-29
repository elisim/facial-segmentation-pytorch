import numpy as np


def closest_color(rgb, colors):
    """
    :param rgb: color in rgb format (tuple)
    :param colors: list of rgb colors
    :return: closest color to `rgb` from `colors`. "closest" determined as minimum euclidean distance
    """
    r, g, b = rgb
    color_diffs = []
    for color in colors:
        cr, cg, cb = color
        color_diff = np.sqrt((r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2)
        color_diffs.append((color_diff, color))
    return min(color_diffs)[1]


def mask2label(mask, colormap):
    """
    The function maps each color in the original
    image to the closest color in the new colormap.

    :param mask: numpy ndarray representing an image in (H, W, 3) format
    :param colormap: dict of rgb color -> label index
    :return: (H, W) ndarray such that each pixel mapped to it's closest color
    """
    height, width, ch = mask.shape

    mask_labels = np.zeros((height, width), dtype=np.float32)
    for h_ in range(height):
        for w_ in range(width):
            color = mask[h_, w_, :]
            color = closest_color(color, list(colormap.keys()))
            mask_labels[h_, w_] = colormap[color]
    return mask_labels


def label2mask(label_img, colormap):
    """
    :param label_img: numpy ndarray of labeled img in shape (H, W)
    :param colormap: dict of rgb color -> label index
    :return: numpy ndarray representing an image in (H, W, 3) format
    """
    label2color = {label: color for color, label in colormap.items()}
    height, width = label_img.shape

    mask = np.zeros((height, width, 3), dtype=np.float32)
    for h_ in range(height):
        for w_ in range(width):
            label_ = label_img[h_, w_]
            mask[h_, w_] = label2color[label_]

    return mask
