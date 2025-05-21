import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE, ADASYN, SMOTENC
from imblearn.under_sampling import RandomUnderSampler
import sys
sys.path.append("../")
from rfoversample import RFOversampler


class ModelTrainer:
    def __init__(self, x_train, y_train, x_test, y_test,
                 random_state=42,
                 categorical_indices=None,
                 categorical_column_names=None,
                 contains_categoricals=False,
                 encoded=False):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.random_state = random_state
        self.model = None

        self.imbalanced_x_train = x_train.copy()
        self.imbalanced_y_train = y_train.copy()

        self.categorical_indices = categorical_indices or []
        self.cat_column_names = categorical_column_names or []
        self.contains_categoricals = contains_categoricals
        self.encoded = encoded

    def reset_training_data(self):
        self.x_train = self.imbalanced_x_train.copy()
        self.y_train = self.imbalanced_y_train.copy()

    def train_and_evaluate(self, method="none", max_depth=None, n_estimators=100):
        self.reset_training_data()

        if len(self.y_train.unique()) < 2:
            return pd.DataFrame()

        if method == "none" or method == "baseline":
            model = RandomForestClassifier(max_depth=max_depth, random_state=self.random_state)

        elif method == "class_weights":
            model = RandomForestClassifier(max_depth=max_depth, class_weight="balanced", random_state=self.random_state)

        elif method in ["smote", "adasyn", "random_undersampling", "rfoversample", "smotenc"]:
            resampler = ResamplingHandler(
                self.x_train, self.y_train,
                random_state=self.random_state,
                categorical_indices=self.categorical_indices,
                cat_column_names=self.cat_column_names,
                contains_categoricals=self.contains_categoricals,
                encoded=self.encoded
            )

            try:
                if method == "smote":
                    self.x_train, self.y_train = resampler.apply_smote()
                elif method == "adasyn":
                    self.x_train, self.y_train = resampler.apply_adasyn()
                elif method == "random_undersampling":
                    self.x_train, self.y_train = resampler.apply_random_undersampling()
                elif method == "rfoversample":
                    self.x_train, self.y_train = resampler.apply_rfoversample()
                elif method == "smotenc":
                    self.x_train, self.y_train = resampler.apply_smotenc()
            except Exception as e:
                print(f"[ERROR] Resampling failed for method: {method} → {e}")
                return pd.DataFrame()

            if len(self.y_train.unique()) < 2:
                return pd.DataFrame()

            model = RandomForestClassifier(max_depth=max_depth, random_state=self.random_state)

        else:
            raise ValueError(f"Invalid method specified: {method}.")

        try:
            model.fit(self.x_train, self.y_train)
            self.model = model

            predictions = model.predict(self.x_test)
            report_dict = classification_report(self.y_test, predictions, output_dict=True)

            if "weighted avg" not in report_dict:
                return pd.DataFrame()
            weighted_metrics = report_dict.get("weighted avg", {})
            if not all(weighted_metrics.get(metric, 0) > 0 for metric in ['precision', 'recall', 'f1-score']):
                return pd.DataFrame()
            
            return pd.DataFrame(report_dict).T

        except Exception as e:
            print(f"[ERROR] Evaluation failed for method: {method} → {e}")
            return pd.DataFrame()


class ResamplingHandler:
    def __init__(self, x_train, y_train, random_state=42,
                 categorical_indices=None,
                 cat_column_names=None,
                 contains_categoricals=False,
                 encoded=False):
        self.x_train = x_train
        self.y_train = y_train
        self.random_state = random_state

        self.categorical_indices = categorical_indices or []
        self.cat_column_names = cat_column_names or []
        self.contains_categoricals = contains_categoricals
        self.encoded = encoded

    def apply_smote(self):
        smote = SMOTE(random_state=self.random_state)
        return smote.fit_resample(self.x_train, self.y_train)

    def apply_adasyn(self):
        adasyn = ADASYN(random_state=self.random_state)
        return adasyn.fit_resample(self.x_train, self.y_train)

    def apply_random_undersampling(self):
        rus = RandomUnderSampler(random_state=self.random_state)
        return rus.fit_resample(self.x_train, self.y_train)

    def apply_smotenc(self):
        if not self.categorical_indices:
            raise ValueError("SMOTENC requires at least one categorical feature.")
        smotenc = SMOTENC(
            categorical_features=self.categorical_indices,
            random_state=self.random_state
        )
        return smotenc.fit_resample(self.x_train, self.y_train)

    def apply_rfoversample(self):
        rfoversampler = RFOversampler(
            self.x_train,
            self.y_train,
            num_samples=3,
            contains_categoricals=self.contains_categoricals,
            encoded=self.encoded,
            cat_cols=self.cat_column_names
        )
        return rfoversampler.fit()
