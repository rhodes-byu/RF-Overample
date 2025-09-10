import pandas as pd
import numpy as np
from sklearn.utils import check_random_state

class ImbalanceHandler:
    """
    Multiclass-aware imbalance inducer using a *pairwise majority-relative cap*.

    For a given target ratio r_pair (passed via `imbalance_ratio`), and majority
    count n_maj (largest class), each non-majority class k is *downsampled only
    if necessary* to:
        t_k = min(n_k, max(F, floor(r_pair * n_maj)))

    - Keeps **all majority** samples (never downsample majority).
    - Applies the same rule for **binary and multiclass**.
    - Guarantees each minority class has at least **F** examples so neighbor-
      based oversamplers (SMOTE/RFOversample) are safe.
    - Does *not* upsample; this stage only induces/class-strengthens imbalance
      by reducing overly-large minority classes. If a minority class already
      has <= cap (or < F), it is left as-is.
    - Any previous `batch_size` or `min_minority_samples` constraints are not
      used; they’re retained in the signature for backward compatibility.

    Parameters
    ----------
    x_train : pd.DataFrame
        Training feature set.
    y_train : pd.Series
        Training labels. The first column name is preserved for return.
    imbalance_ratio : float, default=0.2
        Per-class target ratio r_pair relative to the majority count
        (e.g., 0.10 means each minority class is capped around 10% of majority).
    batch_size : int, deprecated
        Ignored (kept for backward compatibility).
    random_state : int or np.random.RandomState
        RNG seed/instance.
    min_minority_samples : int, deprecated
        Ignored (kept for backward compatibility).
    floor_base : int, default=30
        Per-class floor F; we use F = max(floor_base, K+1) to ensure enough
        neighbors for oversamplers. With K=5, F defaults to 30.
    K : int, default=5
        Neighbor budget used by oversamplers; used only to set the floor.

    Returns
    -------
    (X_resampled, y_resampled) : (pd.DataFrame, pd.Series)
        The downsampled training set.
    """
    def __init__(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        imbalance_ratio: float = 0.2,
        random_state=42,
        floor_base: int = 30,
        K: int = 5,
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.r_pair = float(imbalance_ratio)
        self.random_state = check_random_state(random_state)
        self.F = int(max(floor_base, K + 1))

    def introduce_imbalance(self):
        y = self.y_train
        X = self.x_train
        label_col = y.name if y.name is not None else "_y_"

        class_counts = y.value_counts()
        if class_counts.empty or len(class_counts) < 2:
            return X.copy(), y.copy()

        print(f"[ImbalanceHandler] Original class distribution: {class_counts.to_dict()}")

        # Identify majority (single largest class)
        maj_class = class_counts.idxmax()
        n_maj = int(class_counts.loc[maj_class])

        # Compute per-class cap relative to majority
        cap = max(self.F, int(np.floor(self.r_pair * n_maj)))
        if cap < self.F:
            cap = self.F  # safety; already guaranteed by max() above

        # Build index selection: keep ALL majority; cap each minority independently
        keep_indices = []

        # keep all majority samples
        maj_idx = y[y == maj_class].index
        keep_indices.extend(maj_idx.tolist())

        # per-minority processing
        for cls, n_cls in class_counts.items():
            if cls == maj_class:
                continue

            cls_idx = y[y == cls].index
            n_cls = int(n_cls)

            # If class has fewer than floor F, leave it untouched (don’t make it smaller)
            if n_cls < self.F:
                target = n_cls
            else:
                # cap at r_pair * n_maj, but never below F
                target = min(n_cls, cap)
                target = max(target, self.F)

            # sample without replacement to the target count
            if n_cls <= target:
                chosen = cls_idx.tolist()
            else:
                chosen = self.random_state.choice(cls_idx, size=target, replace=False).tolist()

            keep_indices.extend(chosen)

        # Shuffle the combined indices for stochasticity
        keep_indices = self.random_state.permutation(keep_indices)

        X_resampled = X.loc[keep_indices].reset_index(drop=True)
        y_resampled = y.loc[keep_indices].reset_index(drop=True)

        final_counts = y_resampled.value_counts()
        print(f"[ImbalanceHandler] Pairwise cap = {cap} (F = {self.F}, r_pair = {self.r_pair})")
        print(f"[ImbalanceHandler] Imbalanced class distribution: {final_counts.to_dict()}")

        return X_resampled, y_resampled
