import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

class DatasetPreprocessor:
    def __init__(self, dataset, target_column=None, test_size=0.3, random_state=42):
        
        """
        Initializes the DatasetPreprocessor.

        Args:
        - dataset (pd.DataFrame): The dataset to preprocess.
        - target_column (str, optional): The name of the target column. Defaults to the first column.
        - test_size (float): Proportion of dataset to allocate for testing.
        - random_state (int): Random seed for reproducibility.
        """

        self.dataset = dataset.copy()  
        self.target_column = target_column if target_column else dataset.columns[0]
        self.test_size = test_size
        self.random_state = random_state
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None
        self._prepare_data()

    def _prepare_data(self):
        # Separate features (X) and target (y)
        x = self.dataset.drop(columns=[self.target_column])
        y = self.dataset[self.target_column]

        # Identify and encode categorical columns
        cat_cols = x.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols) > 0:
            x[cat_cols] = x[cat_cols].apply(lambda col: col.astype("category").cat.codes)
            print(f"Encoded categorical columns: {list(cat_cols)}")

        # Train-test split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=self.test_size, random_state=self.random_state
        )

        # Print summary
        print("\nData Preparation Complete")
        print(f"Train-Test Split: Train ({len(self.x_train)}) | Test ({len(self.x_test)})")
        print(f"Feature Columns: {len(x.columns)} | Categorical Encoded: {len(cat_cols)}\n")

class ImbalanceHandler:
    def __init__(self, x_train, y_train, imbalance_ratio=0.2, batch_size=20, random_state=42):
        """
        Initializes the ImbalanceHandler class.

        Args:
            x_train (pd.DataFrame): Training feature set.
            y_train (pd.Series): Training labels.
            imbalance_ratio (float): Target imbalance ratio (default 0.2 for 20% minority).
            batch_size (int): The number of total samples per batch (default 20).
            random_state (int): Random seed for reproducibility.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.imbalance_ratio = imbalance_ratio
        self.batch_size = batch_size
        self.random_state = random_state

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
            return resample(df, replace=False, n_samples=(len(df) // self.batch_size) * self.batch_size, random_state=self.random_state)

        minority_samples = adjust_class_size(minority_samples)
        majority_samples = adjust_class_size(majority_samples)

        print(f"\nAdjusted Sizes - Minority: {len(minority_samples)}, Majority: {len(majority_samples)}")

        # Apply Imbalance Ratio
        minority_per_batch = int(self.batch_size * self.imbalance_ratio)
        majority_per_batch = self.batch_size - minority_per_batch
        max_iterations = min(len(minority_samples) // minority_per_batch, len(majority_samples) // majority_per_batch)

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