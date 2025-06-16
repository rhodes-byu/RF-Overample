import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

class DatasetPreprocessor:
    def __init__(self, dataset, target_column=None, test_size=0.3, random_state=42, 
                 encoding_method="ordinal", method=None, categorical_indices=None):
        """
        Preprocesses a dataset: encodes categorical features and splits into train/test sets.

        Args:
            dataset (pd.DataFrame): Input dataset.
            target_column (str, optional): Name of the target column. Defaults to the first column.
            test_size (float): Proportion of data to allocate for testing.
            random_state (int): Seed for reproducibility.
            encoding_method (str): Encoding strategy: "ordinal" or "onehot".
            method (str): Name of the resampling method (e.g., 'rfoversample', 'smotenc').
            categorical_indices (list of int, optional): Indices of categorical columns.
        """
        self.dataset = dataset.copy()
        self.target_column = target_column or dataset.columns[0]
        self.test_size = test_size
        self.random_state = random_state
        self.encoding_method = encoding_method.lower()
        self.method = method
        self.categorical_indices = categorical_indices

        self.x_train = self.x_test = self.y_train = self.y_test = None
        self.label_encoded_x = None
        self.cat_column_names = []

        self._prepare_data()

    def _prepare_data(self):
        x = self.dataset.drop(columns=[self.target_column])
        y = self.dataset[self.target_column]

        if self.categorical_indices:
            if max(self.categorical_indices) >= x.shape[1]:
                raise ValueError("Categorical index out of range.")
            self.cat_column_names = [x.columns[i] for i in self.categorical_indices]
        else:
            self.cat_column_names = list(x.select_dtypes(include=["object", "category"]).columns)

        if self.method == "smotenc":
            effective_encoding = "ordinal"
        
        elif self.method == "rfoversample":
            effective_encoding = "onehot"
        
        else:
            effective_encoding = self.encoding_method


        if effective_encoding == "ordinal":
            if self.cat_column_names:
                x[self.cat_column_names] = x[self.cat_column_names].apply(lambda col: col.astype("category").cat.codes)
            self.label_encoded_x = x.copy()

        elif effective_encoding == "onehot":
            encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            x_encoded = encoder.fit_transform(x[self.cat_column_names])
            encoded_df = pd.DataFrame(x_encoded, columns=encoder.get_feature_names_out(self.cat_column_names), index=x.index)

            x = x.drop(columns=self.cat_column_names)
            x = pd.concat([x, encoded_df], axis=1)

        else:
            raise ValueError(f"Invalid encoding method '{self.encoding_method}'. Choose 'ordinal' or 'onehot'.")

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=self.test_size, random_state=self.random_state
        )

