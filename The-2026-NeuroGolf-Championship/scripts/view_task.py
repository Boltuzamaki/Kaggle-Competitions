import json

with open("data/task382.json", "r") as f:
    data = json.load(f)

for i, example in enumerate(data["train"]):
    print(f"Train {i}:")
    input_grid = example["input"]
    output_grid = example["output"]
    for r in range(len(input_grid)):
        print(input_grid[r], "   ", output_grid[r] if r < len(output_grid) else "")
    print()

for i, example in enumerate(data["test"]):
    print(f"Test {i}:")
    input_grid = example["input"]
    output_grid = example["output"]
    for r in range(len(input_grid)):
        print(input_grid[r], "   ", output_grid[r] if r < len(output_grid) else "")
    print()
