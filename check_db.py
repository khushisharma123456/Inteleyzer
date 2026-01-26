import sqlite3

conn = sqlite3.connect('pv_backend/instance/pv_database.db')
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("\n=== DATABASE TABLES ===")
print([t[0] for t in tables])

# Check users table
print("\n=== USERS ===")
cursor.execute("SELECT id, email, full_name, role, is_verified FROM users")
users = cursor.fetchall()
if users:
    for u in users:
        print(f"ID: {u[0]}, Email: {u[1]}, Name: {u[2]}, Role: {u[3]}, Verified: {u[4]}")
else:
    print("No users in database")

# Count
cursor.execute("SELECT COUNT(*) FROM users")
print(f"\nTotal users: {cursor.fetchone()[0]}")

conn.close()
