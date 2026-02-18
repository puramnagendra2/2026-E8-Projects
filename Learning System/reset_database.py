import os
import sqlite3
from database import db

# Delete existing database to recreate with new schema
if os.path.exists("learning_assistant.db"):
    os.remove("learning_assistant.db")
    print("Old database removed")

# Reinitialize the database
db.create_tables()
print("New database created with updated schema")

# Add some sample data
from datetime import datetime
import json

# Create a test user
test_users = [
    ("testuser", "password123", "visual", "beginner", ["programming", "data science"]),
    ("learner", "password123", "auditory", "intermediate", ["machine learning", "ai"]),
]

for username, password, style, level, interests in test_users:
    db.create_user(username, password)
    user_id = db.authenticate_user(username, password)
    if user_id:
        db.update_user_profile(user_id, style, level, interests)
        print(f"Created user: {username}")

print("Database setup complete!")