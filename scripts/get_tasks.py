import sqlite3
conn = sqlite3.connect('repairs/tracker.db')
c = conn.cursor()
c.execute("SELECT task FROM tracker WHERE state != 'ours' ORDER BY task ASC")
print([x[0] for x in c.fetchall()][:20])
