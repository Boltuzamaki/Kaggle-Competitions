import csv, json, os

TASK_DIR = r"C:\Users\chand\OneDrive\Desktop\get_a_job\kaggle_competitions\The 2026 NeuroGolf Championship\data"

rows = list(csv.DictReader(open(os.path.join(os.path.dirname(__file__), "cost_profile.csv"))))
same_size_tasks = []
for r in rows:
    t = int(r["task"])
    d = json.load(open(os.path.join(TASK_DIR, f"task{t:03d}.json")))
    all_ex = d["train"] + d["test"] + d["arc-gen"]
    ok = all(len(e["input"]) == len(e["output"]) and len(e["input"][0]) == len(e["output"][0]) for e in all_ex)
    if ok:
        same_size_tasks.append((t, float(r["cost"])))

same_size_tasks.sort(key=lambda x: -x[1])
print("same-size candidate tasks:", len(same_size_tasks), "of", len(rows))
print("top 20 by cost:", same_size_tasks[:20])
