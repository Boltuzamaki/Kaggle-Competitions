import os, glob
import sqlite3

db_path = 'repairs/tracker.db'
conn = sqlite3.connect(db_path)
c = conn.cursor()

c.execute("SELECT task, our_cost FROM tasks WHERE state='ours' AND our_cost BETWEEN 250000 AND 350000")
rows = c.fetchall()
suspicious_tasks = [r[0] for r in rows]

json_files = glob.glob('repairs/*_fails.json')
npy_files = glob.glob('repairs/*_hash_weights.npy')

hack_tasks = set(suspicious_tasks)
for py_file in glob.glob('repairs/user_code/task*.py'):
    with open(py_file, 'r', encoding='utf-8') as f:
        content = f.read()
        if 'W_hash' in content and 'fails_data' in content:
            t = int(os.path.basename(py_file).replace('task', '').replace('.py', ''))
            hack_tasks.add(t)

print('All identified hacked tasks to revert:', sorted(hack_tasks))

for t in hack_tasks:
    c.execute("UPDATE tasks SET state = 'todo', our_cost = NULL, our_points = NULL, n_fail = NULL WHERE task = ?", (t,))
    py_path = f'repairs/user_code/task{t:03d}.py'
    if os.path.exists(py_path):
        os.remove(py_path)
    onnx_path = f'repairs/task{t:03d}.onnx'
    if os.path.exists(onnx_path):
        os.remove(onnx_path)
        
for f in json_files + npy_files:
    os.remove(f)

conn.commit()
conn.close()
