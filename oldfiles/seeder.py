import numpy as np 
import os
import json

def get_or_create_random_seeds(seed_file="random_seeds.json", n_seeds=3, seed=42):
    if os.path.exists(seed_file):
        with open(seed_file, "r") as f:
            return json.load(f)

    np.random.seed(seed)
    random_seeds = np.random.choice(range(1_000, 10_000), size=n_seeds, replace=False).tolist()

    with open(seed_file, "w") as f:
        json.dump(random_seeds, f)

    return random_seeds

if __name__ == "__main__":
    seeds = get_or_create_random_seeds()
    print(f"Generated or loaded seeds: {seeds}")
