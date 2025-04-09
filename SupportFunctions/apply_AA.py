from AA_Fast import AA_Fast
import pandas as pd
import numpy as np

def find_minority_archetypes(X_train, y_train, n_archetypes=10, archetype_proportion=None):
    """
    Identifies archetypal points from the minority class using AA_Fast.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        n_archetypes (int): Number of archetypes if proportion not specified.
        archetype_proportion (float, optional): Proportion of minority samples to use 
                                                (overrides n_archetypes if provided).

    Returns:
        pd.DataFrame: Archetypes extracted from the minority class.
    """
    minority_class = y_train.value_counts().idxmin()
    X_minority = X_train[y_train == minority_class]

    if archetype_proportion is not None:
        if not (0 < archetype_proportion <= 1):
            raise ValueError("archetype_proportion must be between 0 and 1.")
        desired = int(np.round(archetype_proportion * len(X_minority)))
    else:
        desired = n_archetypes

    n_archetypes_final = min(desired, len(X_minority))

    aa = AA_Fast(n_archetypes=n_archetypes_final)
    aa.fit(X_minority.to_numpy())

    return pd.DataFrame(aa.Z, columns=X_minority.columns)

def merge_archetypes_with_minority(X_train, y_train, archetypes, 
                                   sample_percentage=None, sample_number=0, 
                                   random_state=42):
    """
    Combines archetypes with a sample of original minority instances.

    Args:
        X_train (pd.DataFrame): Training features (imbalanced).
        y_train (pd.Series): Training labels.
        archetypes (pd.DataFrame): Archetypal points from the minority class.
        sample_percentage (float, optional): Proportion of original minority class to sample.
        sample_number (int): Fixed number of samples if percentage not used.
        random_state (int): Seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: New training features and labels.
    """
    minority_class = y_train.value_counts().idxmin()
    majority_class = y_train.value_counts().idxmax()

    X_minority = X_train[y_train == minority_class]
    X_majority = X_train[y_train == majority_class]
    y_majority = y_train[y_train == majority_class]

    # Sample from original minority class
    if sample_percentage and sample_percentage > 0:
        X_sampled = X_minority.sample(frac=sample_percentage, random_state=random_state)
    elif sample_number > 0:
        X_sampled = X_minority.sample(n=sample_number, random_state=random_state)
    else:
        X_sampled = pd.DataFrame(columns=X_minority.columns)

    # Combine archetypes and sampled minority points
    X_new_minority = pd.concat([X_sampled, archetypes], ignore_index=True)
    y_new_minority = pd.Series([minority_class] * len(X_new_minority), name=y_train.name)

    # Merge with majority data
    X_new_train = pd.concat([X_majority, X_new_minority], ignore_index=True)
    y_new_train = pd.concat([y_majority, y_new_minority], ignore_index=True)

    # Shuffle combined dataset
    combined = pd.concat([X_new_train, y_new_train], axis=1).sample(
        frac=1, random_state=random_state).reset_index(drop=True)

    X_new_train = combined.drop(columns=[y_new_train.name])
    y_new_train = combined[y_new_train.name]

    return X_new_train, y_new_train
