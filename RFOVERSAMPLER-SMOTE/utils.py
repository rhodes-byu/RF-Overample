from sklearn.utils import resample
import sys
sys.path.append("../SupportFunctions")
sys.path.append('../')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from rfoversample import RFOversampler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from prepare_datasets import DatasetPreprocessor
from utils import introduce_imbalance


def introduce_imbalance(x, y, imbalance_ratio=0.2, random_state=42):
    """
    Downsample all non-majority classes to match the given imbalance ratio.

    Args:
        x (pd.DataFrame): Features.
        y (pd.Series): Labels.
        imbalance_ratio (float): Desired ratio of each minority class to the majority class.
        random_state (int): Random seed.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: The imbalanced x and y.
    """
    # Combine into one DataFrame
    df = pd.concat([x, y], axis=1)
    label_col = y.name

    # Find majority class and its count
    value_counts = y.value_counts()
    maj_label = value_counts.idxmax()
    maj_count = value_counts.max()

    frames = []

    for label, count in value_counts.items():
        class_df = df[df[label_col] == label]

        if label == maj_label:
            frames.append(class_df)  # Keep majority class as-is
        else:
            n_samples = max(1, int(imbalance_ratio * maj_count))
            sampled_df = resample(class_df, replace=True, n_samples=n_samples, random_state=random_state)
            frames.append(sampled_df)

    # Combine and shuffle
    result_df = pd.concat(frames).sample(frac=1, random_state=random_state).reset_index(drop=True)
    x_new = result_df.drop(columns=label_col)
    y_new = result_df[label_col]

    # print(f"Original class distribution: {value_counts.to_dict()}")
    # print(f"Imbalanced class distribution: {y_new.value_counts().to_dict()}")

    return x_new, y_new


def Compare_RF_F1scores(OG_x_train, OG_y_train, RF_x_train_upsampled, RF_y_train_upsampled, SM_x_train_upsampled, SM_y_train_upsampled, x_test, y_test):

    #train, fit, and predict original points
    original_RF = RandomForestClassifier(class_weight='balanced')
    original_RF.fit(OG_x_train, OG_y_train)
    y_pred_original = original_RF.predict(x_test)

    #train, fit, and predict points upsampled by Random Forest Upsampler
    RF_upsampled_RF = RandomForestClassifier()
    RF_upsampled_RF.fit(RF_x_train_upsampled, RF_y_train_upsampled)
    y_pred_RF_upsampled = RF_upsampled_RF.predict(x_test)

    #train, fit, and predict points upsampled by SMOTE
    SM_upsampled_RF = RandomForestClassifier()
    SM_upsampled_RF.fit(SM_x_train_upsampled, SM_y_train_upsampled)
    y_pred_SM_upsampled = SM_upsampled_RF.predict(x_test)

    #append f1 scores
    scores_OG = f1_score(y_test, y_pred_original, average='weighted')
    scores_RF_upsampled = f1_score(y_test, y_pred_RF_upsampled, average='weighted')
    scores_SM_upsampled = f1_score(y_test, y_pred_SM_upsampled, average='weighted')

    return scores_OG, scores_RF_upsampled, scores_SM_upsampled


def run(n, data,target, ratio, categorical=False, encoded=False, cat_col=None):
    OG_scores = np.zeros(n)
    RF_upsampled_scores = np.zeros(n)
    SM_upsampled_scores = np.zeros(n)
    for i in range(n):
        preprocessor = DatasetPreprocessor(data, target_column=target)
        x_train, y_train, x_test, y_test = (preprocessor.x_train, preprocessor.y_train,
                                                    preprocessor.x_test, preprocessor.y_test)

        x_train_imbal, y_train_imbal = introduce_imbalance(x_train, y_train, imbalance_ratio=ratio)

        Oversampler = RFOversampler(x_train=x_train_imbal, y_train=y_train_imbal, contains_categoricals=categorical, encoded=encoded, cat_cols=cat_col)
        RF_upsampled_x_train, RF_upsampled_y_train = Oversampler.fit()

        smote = SMOTE(random_state=42)
        SM_upsampled_x_train, SM_upsampled_y_train = smote.fit_resample(x_train_imbal, y_train_imbal)
    
        OG_score, RF_upsampled_score, SM_upsampled_score = Compare_RF_F1scores( 
        x_train_imbal, 
        y_train_imbal, 
        RF_upsampled_x_train, 
        RF_upsampled_y_train, 
        SM_upsampled_x_train, 
        SM_upsampled_y_train, 
        x_test, 
        y_test
        )
        OG_scores[i] = OG_score
        RF_upsampled_scores[i] = RF_upsampled_score
        SM_upsampled_scores[i] = SM_upsampled_score
    
    print(f"No Upsampling avg f1 score: {np.mean(OG_scores)}")
    print(f"RF avg f1 score: {np.mean(RF_upsampled_scores)}")
    print(f"SMOTE avg f1 score {np.mean(SM_upsampled_scores)}")
    print(np.mean(RF_upsampled_scores) - np.mean(SM_upsampled_scores))
    