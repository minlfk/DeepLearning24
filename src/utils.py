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

def save_list(file_name, data):
    # Extract field names (keys of the dictionaries)
    fieldnames = data[0].keys()
    
    with open(file_name, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # Write header row (field names)
        writer.writeheader()
        
        # Write each dictionary as a row
        writer.writerows(data)

def save_similarities(file_name, data):
    # Transpose the data
    rows = [dict(zip(data.keys(), values)) for values in zip(*data.values())]
    
    # Save to CSV
    with open(file_name, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        writer.writeheader()
        writer.writerows(rows)