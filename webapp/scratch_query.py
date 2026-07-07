import sqlite3
import os

db_path = r'c:\Users\chand\OneDrive\Desktop\get_a_job\kaggle_competitions\The 2026 NeuroGolf Championship\repairs\tracker.db'

try:
    c = sqlite3.connect(db_path)
    # Check what tasks exist first
    cursor = c.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    if ('tasks',) in tables:
        tasks = c.execute("SELECT task FROM tasks WHERE state != 'ours'").fetchall()
        print("Tasks not solved ('ours'):")
        print([t[0] for t in tasks])
    else:
        print("Table 'tasks' not found in database.")
except Exception as e:
    print(f"Error: {e}")
