import pandas as pd
from sklearn.model_selection import train_test_split

class DatasetPreprocessor:
    def __init__(self, dataset, target_column=None, test_size=0.3, random_state=42, 
                 encoding_method="ordinal"):
        """
        Preprocesses a dataset: encodes categorical features and splits into train/test sets.

        Args:
            dataset (pd.DataFrame): Input dataset.
            target_column (str, optional): Name of the target column. Defaults to the first column.
            test_size (float): Proportion of data to allocate for testing.
            random_state (int): Seed for reproducibility.
            encoding_method (str): Encoding strategy: "ordinal" or "onehot".
        """
        self.dataset = dataset.copy()
        self.target_column = target_column or dataset.columns[0]
        self.test_size = test_size
        self.random_state = random_state
        self.encoding_method = encoding_method.lower()

        self.x_train = self.x_test = self.y_train = self.y_test = None
        self.label_encoded_x = None

        self._prepare_data()

    def _prepare_data(self):
        x = self.dataset.drop(columns=[self.target_column])
        y = self.dataset[self.target_column]
        cat_cols = x.select_dtypes(include=["object", "category"]).columns

        # Label encoding always stored for "ordinal"
        self.label_encoded_x = x.copy()
        if not cat_cols.empty:
            self.label_encoded_x[cat_cols] = self.label_encoded_x[cat_cols].apply(
                lambda col: col.astype("category").cat.codes
            )

        # Choose encoding
        if self.encoding_method == "ordinal":
            x = self.label_encoded_x
        elif self.encoding_method == "onehot":
            x = pd.get_dummies(x, columns=cat_cols, drop_first=True)
        else:
            raise ValueError(f"Invalid encoding method '{self.encoding_method}'. Choose 'ordinal' or 'onehot'.")

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=self.test_size, random_state=self.random_state
        )

