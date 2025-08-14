import numpy as np
import pandas as pd
from rfgap import RFGAP


class RFOversampler:

    def __init__(
        self,
        x_train,
        y_train,
        contains_categoricals=False,
        encoded=False,
        cat_cols=None,
        K=7,
        add_noise=True,
        noise_scale=0.15,
        cat_prob_sample=True,
        random_state=None,
        *,
        enforce_domains=True,
        binary_strategy="bernoulli",
        hybrid_perturb_frac=0.0,
    ):
        self.Data = pd.concat([y_train, x_train], axis=1)
        self.target_ind = 0
        self.contains_categoricals = contains_categoricals
        self.encoded = encoded
        self.cat_cols = cat_cols or []
        self.cols = x_train.columns.tolist()

        self.K = max(2, int(K))
        self.add_noise = bool(add_noise)
        self.noise_scale = float(noise_scale)
        self.cat_prob_sample = bool(cat_prob_sample)

        self.enforce_domains = bool(enforce_domains)
        self.binary_strategy = str(binary_strategy)
        self.hybrid_perturb_frac = float(hybrid_perturb_frac)

        self._rng = np.random.default_rng(random_state)

    # helper methods
    @staticmethod
    def _normalize_weights(w: np.ndarray) -> np.ndarray:
        s = float(np.sum(w))
        if s <= 1e-12:
            return np.ones_like(w, dtype=float) / max(len(w), 1)
        return w / s

    @staticmethod
    def _from_dummies_safe(df: pd.DataFrame, sep: str = "_") -> pd.DataFrame:
        """Inverse of get_dummies with a fallback for older pandas versions."""
        try:
            return pd.from_dummies(df, sep=sep)
        except AttributeError:
            groups = {}
            for c in df.columns:
                base = c.split(sep)[0]
                groups.setdefault(base, []).append(c)
            out = {}
            for base, cols in groups.items():
                arr = df[cols].to_numpy()
                idx = arr.argmax(axis=1)
                levels = [cols[i].split(sep, 1)[1] if sep in cols[i] else cols[i] for i in idx]
                out[base] = levels
            return pd.DataFrame(out, index=df.index)

    def _split_numeric_categorical(self, x: pd.DataFrame, cat_dict: dict) -> tuple:
        """Return (x_num, x_cat, encoded_num_cols, encoded_cat_cols).
        Works for both encoded=False (after get_dummies) and encoded=True.
        """
        if not self.contains_categoricals or not cat_dict:
            return x, None, x.columns.tolist(), []

        # Collect one-hot columns for each categorical by name prefix
        encoded_cat_cols = []
        for cat_name, _n_levels in cat_dict.items():
            pref = f"{cat_name}_"
            cols = [c for c in x.columns if c.startswith(pref)]
            # if levels weren't found with exact prefix, fall back to any startswith cat_name
            if not cols:
                cols = [c for c in x.columns if c.startswith(cat_name)]
            encoded_cat_cols.extend(cols)

        encoded_num_cols = [c for c in x.columns if c not in encoded_cat_cols]
        x_numerical = x[encoded_num_cols]
        x_categorical = x[encoded_cat_cols] if encoded_cat_cols else None
        return x_numerical, x_categorical, encoded_num_cols, encoded_cat_cols

    def _infer_cat_dict_after_encoding(self, x: pd.DataFrame) -> dict:
        """Build {base_name: n_levels} after we know the actual dummy columns."""
        cat_dict = {}
        if not self.contains_categoricals:
            return cat_dict
        if not self.encoded:
            # we had original names in self.cat_cols
            for base in self.cat_cols:
                n = sum(1 for c in x.columns if c.startswith(f"{base}_"))
                if n == 0:
                    n = sum(1 for c in x.columns if c.startswith(base))
                if n > 0:
                    cat_dict[base] = n
        else:
            # encoded=True: base names already provided, count columns
            for base in self.cat_cols:
                n = sum(1 for c in x.columns if c.startswith(f"{base}_"))
                if n == 0:
                    n = sum(1 for c in x.columns if c.startswith(base))
                if n > 0:
                    cat_dict[base] = n
        return cat_dict

    def _build_numeric_domain_metadata(self, x_num: pd.DataFrame):
        """Detect binary / integer-like numeric columns and global bounds."""
        num_cols = x_num.columns.tolist()
        mins = x_num.min(axis=0)
        maxs = x_num.max(axis=0)

        def is_binary(col: str) -> bool:
            vals = pd.unique(x_num[col].dropna())
            if len(vals) <= 2:
                s = set(map(float, vals.tolist()))
                return s.issubset({0.0, 1.0})
            return False

        def is_integerish(col: str) -> bool:
            return (
                pd.api.types.is_integer_dtype(x_num[col].dtype)
                or np.allclose(x_num[col].dropna().to_numpy() % 1.0, 0.0)
            )

        binary_mask = {c: is_binary(c) for c in num_cols}
        integer_mask = {c: (not binary_mask[c]) and is_integerish(c) for c in num_cols}
        return num_cols, mins, maxs, binary_mask, integer_mask

    # fit method
    def fit(self):
 
        # Prepare X / y (one-hot if needed)
        if self.contains_categoricals and not self.encoded:
            data_encoded = pd.get_dummies(self.Data, columns=self.cat_cols, dtype=int)
            y = data_encoded.iloc[:, self.target_ind]
            x = data_encoded.drop(self.Data.columns[self.target_ind], axis=1)
        else:
            y = self.Data.iloc[:, self.target_ind]
            x = self.Data.drop(self.Data.columns[self.target_ind], axis=1)

        # Build categorical group sizes from actual columns
        cat_dict = self._infer_cat_dict_after_encoding(x)

        # Split into numeric / categorical by **names**
        x_num, x_cat, num_cols, cat_cols = self._split_numeric_categorical(x, cat_dict)

        # Numeric domain metadata (for optional projection)
        num_col_list, num_mins, num_maxs, binary_mask, integer_mask = self._build_numeric_domain_metadata(x_num)

        # Train RF, get proximities (on ORIGINAL data only)
        rf = RFGAP(y=y, class_weight='balanced', prediction_type='classification', matrix_type='dense')
        rf.fit(x, y)
        prox = rf.get_proximities()  # (n_samples, n_samples)

        # Class accounting (deterministic majority in case of ties)
        vc = y.value_counts()
        # break ties by sorted label order for stability
        maj_label = vc.sort_values(ascending=False).index[0]
        maj_count = int(vc.loc[maj_label])
        classes = list(vc.index)
        class_counts = {c: int(vc.loc[c]) for c in classes}

        y_np = y.to_numpy()
        n_orig = len(y)
        orig_indices_by_class = {c: np.where(y_np == c)[0] for c in classes}
        maj_indices_orig = orig_indices_by_class[maj_label]

        # Collect all synthesized rows; append ONCE at end
        synth_num_rows = []
        synth_cat_rows = [] if (self.contains_categoricals and x_cat is not None and len(cat_cols) > 0) else None
        synth_y = []

        for label in classes:
            if label == maj_label:
                continue
            sample_indices = orig_indices_by_class[label]
            upsample_size = maj_count - class_counts[label]
            if upsample_size <= 0 or len(sample_indices) == 0 or len(maj_indices_orig) == 0:
                continue

            # Boundary-biased seed weights: avg proximity to majority
            avg_prox_to_maj = prox[np.ix_(sample_indices, maj_indices_orig)].mean(axis=1)
            sampling_probs = self._normalize_weights(avg_prox_to_maj)

            for _ in range(upsample_size):
                # 1) Pick a boundary-biased seed among minority samples
                seed = self._rng.choice(sample_indices, p=sampling_probs)

                # 2) Find top-K minority neighbors by RF proximity to seed
                p_seed_to_min = prox[seed, sample_indices]
                order = np.argsort(-p_seed_to_min)[: self.K]
                nbr_idx = sample_indices[order]
                w = self._normalize_weights(p_seed_to_min[order])

                # --- Numeric synthesis ---
                Xn = x_num.iloc[nbr_idx].to_numpy()  # (K, d_num)

                # Option A (hybrid): with some probability, pick closest neighbor and perturb
                use_hybrid = (self.hybrid_perturb_frac > 0.0) and (self._rng.random() < self.hybrid_perturb_frac)
                if use_hybrid:
                    j_closest = int(np.argmax(p_seed_to_min[order]))
                    base_vec = Xn[j_closest].astype(float)
                    new_num = base_vec.copy()
                else:
                    # Convex combination (averaging) to get a local mean point
                    new_num = (w @ Xn).astype(float)

                # local noise
                if self.add_noise and Xn.shape[0] >= 2:
                    mu = Xn.mean(axis=0)
                    S = np.cov((Xn - mu).T) if Xn.shape[0] > 1 else np.eye(Xn.shape[1])
                    S = np.atleast_2d(S)
                    # shrinkage + ridge for stability
                    alpha = 1e-3
                    S = (1.0 - alpha) * S + alpha * np.eye(S.shape[0])
                    S = S + 1e-8 * np.eye(S.shape[0])
                    try:
                        eps = self._rng.multivariate_normal(np.zeros(Xn.shape[1]), self.noise_scale * S)
                        new_num = new_num + eps
                    except np.linalg.LinAlgError:
                        diag = np.clip(np.diag(S), 1e-8, None)
                        eps = self._rng.normal(0.0, np.sqrt(self.noise_scale * diag), size=Xn.shape[1])
                        new_num = new_num + eps

                # --- Domain projection for numeric features (optional) ---
                if self.enforce_domains:
                    # local bounds from neighbors, combined with global bounds
                    local_min = Xn.min(axis=0)
                    local_max = Xn.max(axis=0)
                    mins = np.minimum(local_min, num_mins[num_col_list].to_numpy())
                    maxs = np.maximum(local_max, num_maxs[num_col_list].to_numpy())

                    # 1) clip to plausible range
                    new_num = np.clip(new_num, mins, maxs)

                    # 2) adjust binary & integer-like features
                    for j, col in enumerate(num_col_list):
                        if binary_mask[col]:
                            if self.binary_strategy == "threshold":
                                new_num[j] = 1.0 if new_num[j] >= 0.5 else 0.0
                            else:  # bernoulli using weighted neighbor values
                                p = float(np.clip((w @ Xn[:, j]) / max(w.sum(), 1e-12), 0.0, 1.0))
                                new_num[j] = 1.0 if self._rng.random() < p else 0.0
                        elif integer_mask[col]:
                            val = np.rint(new_num[j])
                            jmin = np.floor(mins[j])
                            jmax = np.ceil(maxs[j])
                            new_num[j] = float(np.clip(val, jmin, jmax))

                synth_num_rows.append(new_num)

                # --- Categorical synthesis (weighted local frequencies) ---
                if self.contains_categoricals and x_cat is not None and cat_cols:
                    # Build one-hot vector per categorical group sequentially
                    new_cat = []
                    start = 0
                    for cat_name, n_levels in cat_dict.items():
                        end = start + n_levels
                        block = x_cat.iloc[nbr_idx, start:end]  # (K, n_levels)
                        B = block.to_numpy()
                        p = w @ B  # weighted frequency per level
                        s = p.sum()
                        if s <= 0:
                            p = np.ones_like(p) / len(p)
                        else:
                            p = p / s
                        if self.cat_prob_sample:
                            idx = int(self._rng.choice(len(p), p=p))
                        else:
                            idx = int(np.argmax(p))
                        one_hot = np.zeros_like(p, dtype=int)
                        one_hot[idx] = 1
                        new_cat.append(one_hot)
                        start = end
                    synth_cat_rows.append(np.concatenate(new_cat, axis=0))

                synth_y.append(label)

        # Nothing to add
        if not synth_num_rows:
            return x, y

        # Concatenate once at the end
        new_num_df = pd.DataFrame(np.vstack(synth_num_rows), columns=num_col_list)
        if synth_cat_rows is not None and len(synth_cat_rows) > 0:
            new_cat_df = pd.DataFrame(np.vstack(synth_cat_rows), columns=cat_cols)
        else:
            new_cat_df = None

        if self.contains_categoricals and new_cat_df is not None:
            if not self.encoded:
                # convert one-hot blocks back to original categories for concat
                new_points_cat_non_dummy = self._from_dummies_safe(new_cat_df, sep="_")
                old_points_cat_non_dummy = self._from_dummies_safe(x_cat, sep="_")
                new_combined_x = pd.concat([new_num_df, new_points_cat_non_dummy], axis=1)
                old_combined_x = pd.concat([x_num, old_points_cat_non_dummy], axis=1)
            else:
                new_combined_x = pd.concat([new_num_df, new_cat_df], axis=1)
                old_combined_x = pd.concat([x_num, x_cat], axis=1)
        else:
            old_combined_x = x
            new_combined_x = new_num_df

        x_out = pd.concat([old_combined_x, new_combined_x], axis=0).reset_index(drop=True)
        y_out = pd.concat([y, pd.Series(np.array(synth_y, dtype=y.dtype))], axis=0).reset_index(drop=True)

        # Try to restore original column order if applicable
        try:
            x_out = x_out.reindex(columns=self.cols, fill_value=np.nan)
        except Exception:
            pass

        return x_out, y_out
