import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DatasetPreprocessor:
    def __init__(self, dataset, target_column=None, test_size=0.3, random_state=42, 
                 encoding_method="ordinal"):
        """
        Initializes the DatasetPreprocessor.

        Args:
        - dataset (pd.DataFrame): The dataset to preprocess.
        - target_column (str, optional): The name of the target column. Defaults to the first column.
        - test_size (float): Proportion of dataset to allocate for testing.
        - random_state (int): Random seed for reproducibility.
        - encoding_method (str): Encoding method for categorical variables. 
                                 Options: "ordinal", "onehot", "frequency", "barycentric".
        """
        self.dataset = dataset.copy()
        self.target_column = target_column if target_column else dataset.columns[0]
        self.test_size = test_size
        self.random_state = random_state
        self.encoding_method = encoding_method.lower()

        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None
        self.label_encoded_x = None  # Stores a label-encoded version for RF
        self.barycentric_x = None  # Stores a barycentric-encoded version for AA
        self.archetypes = None  # Placeholder for archetypes
        self._prepare_data()

    def _prepare_data(self):
        # Separate features (X) and target (y)
        x = self.dataset.drop(columns=[self.target_column])
        y = self.dataset[self.target_column]

        # Identify categorical columns
        cat_cols = x.select_dtypes(include=["object", "category"]).columns

        # Label encoding (always stored as a backup for RF)
        self.label_encoded_x = x.copy()
        if len(cat_cols) > 0:
            self.label_encoded_x[cat_cols] = self.label_encoded_x[cat_cols].apply(
                lambda col: col.astype("category").cat.codes
            )

        # Apply encoding method
        if self.encoding_method == "ordinal":
            # Use label encoding (already stored in self.label_encoded_x)
            x = self.label_encoded_x
            print(f"Ordinal encoded categorical columns: {list(cat_cols)}")

        elif self.encoding_method == "onehot":
            x = pd.get_dummies(x, columns=cat_cols, drop_first=True)
            print(f"One-hot encoded categorical columns: {list(cat_cols)}")
        
        else:
            raise ValueError(f"Invalid encoding method '{self.encoding_method}'. Choose from: 'ordinal', 'onehot', 'frequency', 'barycentric'.")

        # Train-test split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=self.test_size, random_state=self.random_state
        )

        print("\nData Preparation Complete")
        print(f"Train-Test Split: Train ({len(self.x_train)}) | Test ({len(self.x_test)})")
        print(f"Feature Columns: {len(x.columns)} | Categorical Processed: {len(cat_cols)}\n")

    def _barycentric_encode(self, x, cat_cols):
        """
        Applies barycentric encoding to categorical columns.

        Args:
        - x (pd.DataFrame): The feature matrix containing categorical columns.
        - cat_cols (list): List of categorical column names.

        Returns:
        - pd.DataFrame: Barycentric encoded dataset with expanded features.
        """
        x_bary = x.copy()
        for col in cat_cols:
            categories = x[col].astype("category").cat.categories
            n_categories = len(categories)

            # Construct a simplex representation (n_categories -> n_categories - 1 dimensions)
            simplex_dim = n_categories - 1
            barycentric_coords = np.eye(n_categories, simplex_dim, dtype=np.float64) if simplex_dim > 0 else np.zeros((n_categories, 1))

            # Map categories to barycentric coordinates
            mapping = {cat: barycentric_coords[i] for i, cat in enumerate(categories)}

            # Create separate columns for each dimension of the barycentric encoding
            encoded_values = x[col].map(mapping).apply(pd.Series)
            encoded_values.columns = [f"{col}_bary_{i}" for i in range(simplex_dim)]
            
            # Drop original categorical column and replace it with encoded dimensions
            x_bary = x_bary.drop(columns=[col]).join(encoded_values)

        return x_bary
