from AA_Fast import AA_Fast
import pandas as pd
import numpy as np

def find_minority_archetypes(X_train, y_train, n_archetypes=10, archetype_proportion=None):
    """
    Extracts archetypal points from the minority class in the training set using AA_Fast.

    This function separates the minority class from the majority class and then
    determines the number of archetypes based on either a specified proportion of the
    minority samples or a discrete number (defaulting to 10). If the desired number of
    archetypes exceeds the available minority data points, it defaults to the number of
    minority samples.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        n_archetypes (int, optional): Discrete number of archetypes if archetype_proportion 
                                      is not provided. Defaults to 10.
        archetype_proportion (float, optional): Proportion (0 < value ≤ 1) of the minority
                                                class to use as the number of archetypes.
                                                If provided, this value is used to compute
                                                the desired number of archetypes.

    Returns:
        pd.DataFrame: DataFrame containing the archetypal points for the minority class.
    """
    # Identify the minority class label (the one with the fewest samples)
    minority_class = y_train.value_counts().idxmin()
    X_minority = X_train[y_train == minority_class]

    # Determine the desired number of archetypes
    if archetype_proportion is not None:
        if not (0 < archetype_proportion <= 1):
            raise ValueError("archetype_proportion must be between 0 and 1.")
        desired_archetypes = int(np.round(archetype_proportion * len(X_minority)))
    else:
        desired_archetypes = n_archetypes

    # Ensure we don't request more archetypes than available minority samples
    effective_n_archetypes = min(desired_archetypes, len(X_minority))
    print(f"Using {effective_n_archetypes} archetypes for the minority class.")

    # Convert the minority features to a numpy array for AA_Fast
    X_minority_array = X_minority.to_numpy()

    # Run archetypal analysis on the minority samples
    aa = AA_Fast(n_archetypes=effective_n_archetypes)
    aa.fit(X_minority_array)

    # Extract archetypes (assumed to be stored in aa.Z) and convert back to a DataFrame
    archetypes = pd.DataFrame(aa.Z, columns=X_minority.columns)
    
    return archetypes

def merge_archetypes_with_minority(X_train, y_train, archetypes, 
                                   sample_percentage=None, sample_number=0, 
                                   random_state=42):
    """
    Merge archetypal points with a random sample of original minority samples 
    from the imbalanced training set.
    
    Args:
        X_train (pd.DataFrame): Original training features (imbalanced).
        y_train (pd.Series): Original training labels.
        archetypes (pd.DataFrame): Archetypal points extracted from the minority class.
        sample_percentage (float, optional): Fraction (0 < value ≤ 1) of the original 
                                             minority class to randomly sample.
        sample_number (int, optional): Fixed number of original minority samples to include 
                                       if sample_percentage is not provided (default=0).
        random_state (int): Random seed for reproducibility.
    
    Returns:
        X_new_train (pd.DataFrame): Updated training features (majority + new minority).
        y_new_train (pd.Series): Updated training labels.
    """
    # Identify minority class label (the one with the fewest samples)
    minority_class = y_train.value_counts().idxmin()
    
    # Extract minority samples from the original training set
    X_minority = X_train[y_train == minority_class]
    
    # Sample from the minority class
    if sample_percentage is not None and sample_percentage > 0:
        X_sampled = X_minority.sample(frac=sample_percentage, random_state=random_state)
    elif sample_number > 0:
        X_sampled = X_minority.sample(n=sample_number, random_state=random_state)
    else:
        # If no sampling is requested, use an empty DataFrame so that only archetypes are used
        X_sampled = pd.DataFrame(columns=X_minority.columns)
    
    # Merge the sampled original minority points with the archetypes
    X_new_minority = pd.concat([X_sampled, archetypes], ignore_index=True)
    y_new_minority = pd.Series([minority_class] * len(X_new_minority), name=y_train.name)
    
    # Extract majority samples (the rest of the training set)
    majority_class = y_train.value_counts().idxmax()
    X_majority = X_train[y_train == majority_class]
    y_majority = y_train[y_train == majority_class]
    
    # Merge the new minority portion with the majority samples
    X_new_train = pd.concat([X_majority, X_new_minority], ignore_index=True)
    y_new_train = pd.concat([y_majority, y_new_minority], ignore_index=True)
    
    # Optionally, shuffle the new training set to mix the samples
    combined = pd.concat([X_new_train, y_new_train], axis=1)
    combined = combined.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Separate features and labels again
    y_col = y_train.name if y_train.name in combined.columns else combined.columns[-1]
    X_new_train = combined.drop(columns=[y_col])
    y_new_train = combined[y_col]
    
    return X_new_train, y_new_train
