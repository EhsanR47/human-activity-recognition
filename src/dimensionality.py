"""Small experiments to show the curse of dimensionality."""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist


def distance_contrast(dimensions, n_points: int = 500, random_state: int = 0) -> pd.DataFrame:
    """Relative contrast between the farthest and nearest pairs of random points.

    contrast = (max distance - min distance) / min distance

    In high dimensions, all points become almost equally far from each other,
    so the contrast goes towards 0. Then "nearest neighbour" loses its meaning,
    which hurts distance-based methods like kNN.
    """
    rng = np.random.default_rng(random_state)
    rows = []
    for d in dimensions:
        points = rng.uniform(size=(n_points, d))
        distances = pdist(points)
        rows.append({
            "dimensions": d,
            "contrast": (distances.max() - distances.min()) / distances.min(),
            "mean_distance": distances.mean(),
            "std_over_mean": distances.std() / distances.mean(),
        })
    return pd.DataFrame(rows)
