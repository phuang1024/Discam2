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


def apply_edge_weights(bbox, edges, aspect, velocity, min_size=20):
    """
    Apply model prediction to bbox, keeping aspect ratio.

    Algorithm:
    First, shift edges by velocity * edge_value.
    Then, adjust bbox to keep aspect ratio.
      Keep center fixed, and adjust width or height to match aspect,
      while maintaining area.

    w * h = A
    w / h = aspect
    w = sqrt(A * aspect)
    h = A / w

    bbox: (x1, y1, x2, y2)
    edges: (up, right, down, left) in [-1, 1]
    aspect: Target aspect ratio (width / height).
    velocity: Max pixels to move per step.
    return: New (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = bbox
    x1 -= velocity * edges[3]
    x2 += velocity * edges[1]
    y1 -= velocity * edges[0]
    y2 += velocity * edges[2]

    # Check negativity and min area.
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    if y2 < y1 + min_size:
        y2 = y1 + min_size
    if x2 < x1 + min_size:
        x2 = x1 + min_size

    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    area = (x2 - x1) * (y2 - y1)

    new_w = (area * aspect) ** 0.5
    new_h = area / new_w

    x1 = center[0] - new_w / 2
    x2 = center[0] + new_w / 2
    y1 = center[1] - new_h / 2
    y2 = center[1] + new_h / 2

    return (x1, y1, x2, y2)


def check_bbox_bounds(bbox, res):
    """
    Ensure bbox is within and smaller than video frame.
    Maintains original bbox size (unless bbox is larger than frame).

    bbox: Original bbox.
    res: (width, height).
    return: New bbox of same shape (if possible) within frame.
    """
    x1, y1, x2, y2 = bbox

    if x2 - x1 > res[0]:
        return (0, 0, res[0], res[1])

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if x2 > res[0]:
        x1 -= (x2 - res[0])
        x2 = res[0]
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if y2 > res[1]:
        y1 -= (y2 - res[1])
        y2 = res[1]

    return (x1, y1, x2, y2)
