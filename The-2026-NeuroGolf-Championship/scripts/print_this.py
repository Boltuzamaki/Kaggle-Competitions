import json
import sys

with open("data/task367.json") as f:
    task = json.load(f)

for i, t in enumerate(task["train"]):
    print(f"--- Train {i} ---")
    print("Input:")
    for row in t["input"]:
        print("".join(str(x) if x != 0 else "." for x in row))
    print("\nOutput:")
    for row in t["output"]:
        print("".join(str(x) if x != 0 else "." for x in row))
    print()

for i, t in enumerate(task["test"]):
    print(f"--- Test {i} ---")
    print("Input:")
    for row in t["input"]:
        print("".join(str(x) if x != 0 else "." for x in row))
    print("\nOutput:")
    for row in t["output"]:
        print("".join(str(x) if x != 0 else "." for x in row))
    print()
