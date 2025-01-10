import datasets
import random
import csv

def subset_ds(ds, ratio):
    if isinstance(ds, datasets.Dataset):
        # Shuffle and subset Hugging Face dataset
        shuffled_dataset = ds.shuffle(seed=42)
        to_keep = int(len(shuffled_dataset) * ratio)
        subset = shuffled_dataset.select(range(to_keep))
    elif isinstance(ds, list):
        # Shuffle and subset Python list
        random.seed(42)
        random.shuffle(ds)
        to_keep = int(len(ds) * ratio)
        subset = ds[:to_keep]
    else:
        raise TypeError("Input should be either a Hugging Face Dataset or a Python list.")

    return subset

def save_list(name, list):
    with open(name, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(list)