import matplotlib.pyplot as plt
import csv

csv_path = "/Users/nathan/code/inf1007/data/penguins_size.csv"

penguins_list = []

with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            penguins_list.append({k:v for k,v in row.items()})

## Plot data
print()
