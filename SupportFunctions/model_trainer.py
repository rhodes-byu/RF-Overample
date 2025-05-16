import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
import sys
sys.path.append("../")
from rfoversample import RFOversampler

class ModelTrainer:
    def __init__(self, x_train, y_train, x_test, y_test, random_state=42):
        """
        Initializes the ModelTrainer.

        Args:
            x_train (pd.DataFrame): Training feature set.
            y_train (pd.Series): Training labels.
            x_test (pd.DataFrame): Testing feature set.
            y_test (pd.Series): Testing labels.
            random_state (int): Random seed for reproducibility.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.random_state = random_state
        self.model = None

        # Store imbalanced data for reset function
        self.imbalanced_x_train = x_train.copy()
        self.imbalanced_y_train = y_train.copy()

    def reset_training_data(self):
        "resets training data to imbalanced state"
        self.x_train = self.imbalanced_x_train.copy()
        self.y_train = self.imbalanced_y_train.copy()

    def train_and_evaluate(self, method="none", max_depth=5, n_estimators=100):
        """
        Trains and evaluates a model based on the specified resampling method.

        Args:
            method (str): Resampling method (options: "none", "class_weights", "smote", "adasyn", "random_undersampling", "easy_ensemble", "rfoversample").
            max_depth (int): Max depth for RandomForestClassifier.
            n_estimators (int): Number of estimators for EasyEnsembleClassifier.

        Returns:
            pd.DataFrame: A classification report formatted for visualization.
        """
        self.reset_training_data()

        print(f"\n[INFO] Training with method: {method.upper()}")
        print("Class distribution before training:\n", self.y_train.value_counts(normalize=True))

        if method == "none":
            model = RandomForestClassifier(max_depth=max_depth, random_state=self.random_state)

        elif method == "class_weights":
            model = RandomForestClassifier(max_depth=max_depth, class_weight="balanced", random_state=self.random_state)

        elif method in ["smote", "adasyn", "random_undersampling", "rfoversample"]:
            resampler = ResamplingHandler(self.x_train, self.y_train, random_state=self.random_state)

            if method == "smote":
                self.x_train, self.y_train = resampler.apply_smote()
            elif method == "adasyn":
                self.x_train, self.y_train = resampler.apply_adasyn()
            elif method == "random_undersampling":
                self.x_train, self.y_train = resampler.apply_random_undersampling()
            elif method == "rfoversample":
                self.x_train, self.y_train = resampler.apply_rfoversample()

            model = RandomForestClassifier(max_depth=max_depth, random_state=self.random_state)

        else:
            raise ValueError(f"Invalid method specified: {method}. Choose from ['none', 'class_weights', 'smote', 'adasyn', 'random_undersampling', 'easy_ensemble', 'rfoversample'].")

        model.fit(self.x_train, self.y_train)
        self.model = model  

        predictions = model.predict(self.x_test)

        report_dict = classification_report(self.y_test, predictions, output_dict=True)
        report_df = pd.DataFrame(report_dict).T

        print("\n[INFO] Model Evaluation Complete")
        return report_df


class ResamplingHandler:
    def __init__(self, x_train, y_train, random_state=42):
        self.x_train = x_train
        self.y_train = y_train
        self.random_state = random_state

    def apply_smote(self):
        """Applies SMOTE and returns the resampled dataset."""
        smote = SMOTE(random_state=self.random_state)
        x_resampled, y_resampled = smote.fit_resample(self.x_train, self.y_train)
        return x_resampled, y_resampled

    def apply_adasyn(self):
        """Applies ADASYN and returns the resampled dataset."""
        adasyn = ADASYN(random_state=self.random_state)
        x_resampled, y_resampled = adasyn.fit_resample(self.x_train, self.y_train)
        return x_resampled, y_resampled

    def apply_random_undersampling(self):
        """Applies Random Undersampling and returns the resampled dataset."""
        rus = RandomUnderSampler(random_state=self.random_state)
        x_resampled, y_resampled = rus.fit_resample(self.x_train, self.y_train)
        return x_resampled, y_resampled

    def apply_rfoversample(self):
        """Applies RF Oversampling and returns the resampled dataset."""
        rfoversampler = RFOversampler(self.x_train, self.y_train, num_samples=3, contains_categoricals=False, encoded=False, cat_cols=None)
        x_resampled, y_resampled = rfoversampler.fit()
        return x_resampled, y_resampled