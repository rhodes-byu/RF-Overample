import pandas as pd
from sklearn.utils import resample, check_random_state

class ImbalanceHandler:
    def __init__(self, x_train, y_train, imbalance_ratio=0.2, batch_size=20, random_state=42):
        """
        Handles creation of an imbalanced dataset with a given ratio and batch constraints.

        Args:
            x_train (pd.DataFrame): Training feature set.
            y_train (pd.Series): Corresponding labels.
            imbalance_ratio (float): Desired minority class ratio per batch (e.g., 0.2 for 20%).
            batch_size (int): Total size of each resampled batch.
            random_state (int or RandomState): Random seed or instance.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.imbalance_ratio = imbalance_ratio
        self.batch_size = batch_size
        self.random_state = check_random_state(random_state)

    def introduce_imbalance(self):
        """
        Resamples the dataset to achieve the specified class imbalance across fixed-size batches.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and labels of the imbalanced training set.
        """
        # Combine features and labels
        full_df = pd.concat([self.x_train, self.y_train.to_frame()], axis=1)
        label_col = self.y_train.name

        # Identify class splits
        class_counts = self.y_train.value_counts()
        minority_class = class_counts.idxmin()
        majority_class = class_counts.idxmax()

        minority_df = full_df[full_df[label_col] == minority_class]
        majority_df = full_df[full_df[label_col] == majority_class]

        # Helper to trim class size to multiple of batch_size
        def trim_to_batches(df):
            usable_n = (len(df) // self.batch_size) * self.batch_size
            return resample(df, replace=False, n_samples=usable_n, random_state=self.random_state)

        minority_df = trim_to_batches(minority_df)
        majority_df = trim_to_batches(majority_df)

        # Calculate per-batch class counts
        min_per_batch = int(self.batch_size * self.imbalance_ratio)
        maj_per_batch = self.batch_size - min_per_batch

        # Determine how many complete batches we can form
        n_batches = min(len(minority_df) // min_per_batch, len(majority_df) // maj_per_batch)

        # Create batches
        batches = [
            pd.concat([
                resample(minority_df, n_samples=min_per_batch, replace=False, random_state=self.random_state),
                resample(majority_df, n_samples=maj_per_batch, replace=False, random_state=self.random_state)
            ])
            for _ in range(n_batches)
        ]

        imbalanced_df = pd.concat(batches).sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        X_resampled = imbalanced_df.drop(columns=[label_col])
        y_resampled = imbalanced_df[label_col]
        return X_resampled, y_resampled
    