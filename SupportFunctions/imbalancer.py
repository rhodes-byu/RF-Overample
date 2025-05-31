import pandas as pd
from sklearn.utils import check_random_state, resample

class ImbalanceHandler:
    def __init__(self, x_train, y_train, imbalance_ratio=0.2, batch_size=100, random_state=42, min_minority_samples=20):
        """
        Handles creation of an imbalanced dataset by resampling with replacement.

        Args:
            x_train (pd.DataFrame): Training feature set.
            y_train (pd.Series): Corresponding labels.
            imbalance_ratio (float): Desired minority class ratio per batch (e.g., 0.2 for 20%).
            batch_size (int): Total size of each resampled batch.
            random_state (int or RandomState): Random seed or instance.
            min_minority_samples (int): Minimum number of samples to retain for the minority class.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.imbalance_ratio = imbalance_ratio
        self.batch_size = batch_size
        self.random_state = check_random_state(random_state)
        self.min_minority_samples = min_minority_samples

    def introduce_imbalance(self):
        """
        Introduces class imbalance by manually resampling with replacement.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and labels of the imbalanced training set.
        """
        y = self.y_train
        X = self.x_train
        full_df = pd.concat([X, y], axis=1)
        label_col = y.name
        class_counts = y.value_counts()

        if class_counts.empty or len(class_counts) < 2:
            return X.copy(), y.copy()

        print(f"Original class distribution: {class_counts.to_dict()}")

        sorted_classes = class_counts.sort_values().index.tolist()
        minority_class = sorted_classes[0]
        other_classes = [cls for cls in sorted_classes if cls != minority_class]

        # Determine number of minority samples
        min_samples = max(int(self.batch_size * self.imbalance_ratio), self.min_minority_samples)
        samples_remaining = self.batch_size - min_samples
        per_other_class = samples_remaining // len(other_classes)

        sampled_frames = []

        # Resample minority class
        minority_df = full_df[full_df[label_col] == minority_class]
        sampled_frames.append(resample(minority_df, replace=True, n_samples=min_samples, random_state=self.random_state))

        # Resample other classes
        for cls in other_classes:
            cls_df = full_df[full_df[label_col] == cls]
            sampled_frames.append(resample(cls_df, replace=True, n_samples=per_other_class, random_state=self.random_state))

        result_df = pd.concat(sampled_frames, axis=0).sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        X_resampled = result_df.drop(columns=[label_col])
        y_resampled = result_df[label_col]

        final_counts = y_resampled.value_counts()
        print(f"Imbalanced class distribution: {final_counts.to_dict()}")

        return X_resampled, y_resampled
