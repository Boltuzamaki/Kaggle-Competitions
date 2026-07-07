"""
Batch-process unsolved tasks: run process_tasks.py for each todo task and report results.
"""
import sqlite3, subprocess, sys, os

PROJ = r'c:\Users\chand\OneDrive\Desktop\get_a_job\kaggle_competitions\The 2026 NeuroGolf Championship'
os.environ['PROJECT_DIR'] = PROJ

conn = sqlite3.connect(os.path.join(PROJ, 'repairs', 'tracker.db'))
c = conn.cursor()
c.execute("SELECT task FROM tasks WHERE state = 'todo' AND task >= 360 AND task <= 370 ORDER BY task")
todo = [r[0] for r in c.fetchall()]
conn.close()

print(f'Tasks to process: {len(todo)}')
print(f'First 10: {todo[:10]}')

# Process each one
for task_id in todo:
    print(f'\n--- Processing task {task_id} ---')
    result = subprocess.run(
        ['python', os.path.join(PROJ, 'webapp', 'process_tasks.py'), str(task_id)],
        cwd=PROJ, capture_output=True, text=True, timeout=120
    )
    output = result.stdout + result.stderr
    # Find the result line
    for line in output.split('\n'):
        if 'Task' in line or 'PASS' in line or 'FAIL' in line or 'ERROR' in line:
            print(f'  {line.strip()}')
    sys.stdout.flush()
