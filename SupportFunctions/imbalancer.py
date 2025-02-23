import pandas as pd
from sklearn.utils import resample, check_random_state

class ImbalanceHandler:
    def __init__(self, x_train, y_train, imbalance_ratio=0.2, batch_size=20, random_state=42):
        """
        Initializes the ImbalanceHandler class.

        Args:
            x_train (pd.DataFrame): Training feature set.
            y_train (pd.Series): Training labels.
            imbalance_ratio (float): Target imbalance ratio (default 0.2 for 20% minority).
            batch_size (int): The number of total samples per batch (default 20).
            random_state (int or RandomState): Random seed or RandomState instance for reproducibility.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.imbalance_ratio = imbalance_ratio
        self.batch_size = batch_size
        self.random_state = check_random_state(random_state)

    def introduce_imbalance(self):
        """Introduces class imbalance while ensuring divisibility by batch_size."""
        print(f"\nApplying imbalance ratio: {self.imbalance_ratio:.2f}")

        # Combine features & labels
        train_df = pd.concat([self.x_train, self.y_train.to_frame()], axis=1)

        # Get class distributions
        class_counts = self.y_train.value_counts()
        print(f"\nOriginal Class Distribution:\n{class_counts.to_string()}")

        # Identify majority and minority class
        minority_class = class_counts.idxmin()
        majority_class = class_counts.idxmax()

        # Separate the classes
        minority_samples = train_df[train_df[self.y_train.name] == minority_class]
        majority_samples = train_df[train_df[self.y_train.name] == majority_class]

        # Adjust class sizes to be multiples of batch_size
        def adjust_class_size(df):
            return resample(
                df,
                replace=False,
                n_samples=(len(df) // self.batch_size) * self.batch_size,
                random_state=self.random_state
            )

        minority_samples = adjust_class_size(minority_samples)
        majority_samples = adjust_class_size(majority_samples)

        print(f"\nAdjusted Sizes - Minority: {len(minority_samples)}, Majority: {len(majority_samples)}")

        # Apply Imbalance Ratio
        minority_per_batch = int(self.batch_size * self.imbalance_ratio)
        majority_per_batch = self.batch_size - minority_per_batch
        max_iterations = min(len(minority_samples) // minority_per_batch,
                             len(majority_samples) // majority_per_batch)

        final_samples = [
            pd.concat([
                resample(minority_samples, replace=False, n_samples=minority_per_batch, random_state=self.random_state),
                resample(majority_samples, replace=False, n_samples=majority_per_batch, random_state=self.random_state)
            ])
            for _ in range(max_iterations)
        ]

        # Combine and shuffle
        imbalanced_train_df = pd.concat(final_samples).sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        print(f"\nNew Class Distribution:\n{imbalanced_train_df[self.y_train.name].value_counts().to_string()}")
        return imbalanced_train_df.drop(columns=[self.y_train.name]), imbalanced_train_df[self.y_train.name]
