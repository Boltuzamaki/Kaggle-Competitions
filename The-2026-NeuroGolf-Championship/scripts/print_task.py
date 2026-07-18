import json

with open("c:/Users/chand/OneDrive/Desktop/get_a_job/kaggle_competitions/The 2026 NeuroGolf Championship/data/task383.json", "r") as f:
    data = json.load(f)

def print_grid(grid, title):
    print(title)
    for row in grid:
        print("".join(str(x) if x != 0 else "." for x in row))
    print()

for i, example in enumerate(data['train']):
    print(f"--- Train {i} ---")
    print_grid(example['input'], "Input:")
    print_grid(example['output'], "Output:")

for i, example in enumerate(data['test']):
    print(f"--- Test {i} ---")
    print_grid(example['input'], "Input:")
    print_grid(example['output'], "Output:")
