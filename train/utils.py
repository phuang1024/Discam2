"""
Utilities.
"""


def expand_bbox_to_aspect(bbox, aspect):
    """
    Expand either width or height w.r.t. center to achieve desired aspect ratio.

    bbox: (x1, y1, x2, y2).
    aspect: Desired width / height ratio.
    """
    x1, y1, x2, y2 = bbox

    curr_aspect = (x2 - x1) / (y2 - y1)

    if curr_aspect > aspect:
        # Expand Y.
        center = (y2 + y1) / 2
        target = (x2 - x1) / aspect
        y1 = center - target / 2
        y2 = center + target / 2

    else:
        # Expand X.
        center = (x2 + x1) / 2
        target = (y2 - y1) * aspect
        x1 = center - target / 2
        x2 = center + target / 2

    return (x1, y1, x2, y2)
