import sqlite3
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
import hashlib

load_dotenv()

class DatabaseManager:
    def __init__(self, db_path="learning_assistant.db"):
        self.db_path = db_path
        self.create_tables()
    
    def get_connection(self):
        """Create and return a database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def create_tables(self):
        """Create all necessary tables if they don't exist"""
        tables_sql = [
            '''CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT,
                created_at TEXT NOT NULL
            )''',
            
            '''CREATE TABLE IF NOT EXISTS user_profiles (
                user_id INTEGER PRIMARY KEY,
                learning_style TEXT DEFAULT 'visual',
                knowledge_level TEXT DEFAULT 'beginner',
                interests TEXT DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )''',
            
            '''CREATE TABLE IF NOT EXISTS learning_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                topic TEXT NOT NULL,
                subtopic TEXT,
                proficiency_score INTEGER DEFAULT 0,
                questions_attempted INTEGER DEFAULT 0,
                questions_correct INTEGER DEFAULT 0,
                last_practiced TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )''',
            
            '''CREATE TABLE IF NOT EXISTS learning_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                topic TEXT NOT NULL,
                subtopic TEXT,
                session_data TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )''',
            
            '''CREATE TABLE IF NOT EXISTS saved_explanations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                topic TEXT NOT NULL,
                explanation TEXT NOT NULL,
                tags TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )''',
            
            '''CREATE TABLE IF NOT EXISTS quiz_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                topic TEXT NOT NULL,
                question TEXT NOT NULL,
                user_answer TEXT,
                correct_answer TEXT,
                is_correct INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )''',
            
            '''CREATE TABLE IF NOT EXISTS learning_resources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                url TEXT,
                topic TEXT NOT NULL,
                difficulty TEXT DEFAULT 'beginner',
                content_type TEXT DEFAULT 'article',
                tags TEXT,
                created_at TEXT NOT NULL
            )'''
        ]
        
        for sql in tables_sql:
            try:
                conn = self.get_connection()
                cursor = conn.cursor()
                cursor.execute(sql)
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"Error creating table: {e}")
    
    def _hash_password(self, password: str) -> str:
        """Hash password for storage"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, username: str, password: str, email: str = None) -> bool:
        """Create a new user"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            password_hash = self._hash_password(password)
            created_at = datetime.now().isoformat()
            
            cursor.execute('''
                INSERT INTO users (username, password_hash, email, created_at)
                VALUES (?, ?, ?, ?)
            ''', (username, password_hash, email, created_at))
            
            user_id = cursor.lastrowid
            
            # Create default profile
            cursor.execute('''
                INSERT INTO user_profiles (user_id, created_at, updated_at)
                VALUES (?, ?, ?)
            ''', (user_id, created_at, created_at))
            
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            return False  # Username already exists
        except Exception as e:
            print(f"Error creating user: {e}")
            return False
    
    def authenticate_user(self, username: str, password: str) -> Optional[int]:
        """Authenticate user and return user_id if successful"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            password_hash = self._hash_password(password)
            
            cursor.execute('''
                SELECT id FROM users 
                WHERE username = ? AND password_hash = ?
            ''', (username, password_hash))
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result else None
        except Exception as e:
            print(f"Error authenticating user: {e}")
            return None
    
    def get_user_profile(self, user_id: int) -> Dict:
        """Get user profile information"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT u.username, u.email, up.learning_style, up.knowledge_level, up.interests
                FROM users u
                LEFT JOIN user_profiles up ON u.id = up.user_id
                WHERE u.id = ?
            ''', (user_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'user_id': user_id,
                    'username': result[0],
                    'email': result[1],
                    'learning_style': result[2] or 'visual',
                    'knowledge_level': result[3] or 'beginner',
                    'interests': json.loads(result[4]) if result[4] else []
                }
            return None
        except Exception as e:
            print(f"Error getting user profile: {e}")
            return None
    
    def update_user_profile(self, user_id: int, learning_style: str = None, 
                          knowledge_level: str = None, interests: List[str] = None):
        """Update user profile"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            update_fields = []
            params = []
            
            if learning_style:
                update_fields.append("learning_style = ?")
                params.append(learning_style)
            
            if knowledge_level:
                update_fields.append("knowledge_level = ?")
                params.append(knowledge_level)
            
            if interests is not None:
                update_fields.append("interests = ?")
                params.append(json.dumps(interests))
            
            if update_fields:
                update_fields.append("updated_at = ?")
                params.append(datetime.now().isoformat())
                params.append(user_id)
                
                cursor.execute(f'''
                    UPDATE user_profiles 
                    SET {', '.join(update_fields)}
                    WHERE user_id = ?
                ''', params)
                
                conn.commit()
            
            conn.close()
            return True
        except Exception as e:
            print(f"Error updating profile: {e}")
            return False
    
    def save_learning_session(self, user_id: int, topic: str, subtopic: str, session_data: Dict):
        """Save a learning session"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO learning_sessions (user_id, topic, subtopic, session_data, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                user_id,
                topic,
                subtopic,
                json.dumps(session_data),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error saving learning session: {e}")
            return False
    
    def get_recent_sessions(self, user_id: int, limit: int = 10) -> List[Dict]:
        """Get recent learning sessions for a user"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, topic, subtopic, session_data, created_at
                FROM learning_sessions
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            ''', (user_id, limit))
            
            rows = cursor.fetchall()
            sessions = []
            
            for row in rows:
                sessions.append({
                    'id': row[0],
                    'topic': row[1],
                    'subtopic': row[2],
                    'session_data': json.loads(row[3]) if row[3] else {},
                    'created_at': row[4]
                })
            
            conn.close()
            return sessions
        except Exception as e:
            print(f"Error getting recent sessions: {e}")
            return []
    
    def update_progress(self, user_id: int, topic: str, correct: int, total_questions: int):
        """Update user progress for a topic"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Check if progress exists for this topic
            cursor.execute('''
                SELECT id, questions_attempted, questions_correct
                FROM learning_progress
                WHERE user_id = ? AND topic = ?
            ''', (user_id, topic))
            
            result = cursor.fetchone()
            
            now = datetime.now().isoformat()
            
            if result:
                # Update existing progress
                new_attempted = result[1] + total_questions
                new_correct = result[2] + correct
                new_proficiency = int((new_correct / new_attempted) * 100) if new_attempted > 0 else 0
                
                cursor.execute('''
                    UPDATE learning_progress
                    SET questions_attempted = ?,
                        questions_correct = ?,
                        proficiency_score = ?,
                        last_practiced = ?
                    WHERE id = ?
                ''', (new_attempted, new_correct, new_proficiency, now, result[0]))
            else:
                # Create new progress record
                proficiency = int((correct / total_questions) * 100) if total_questions > 0 else 0
                
                cursor.execute('''
                    INSERT INTO learning_progress 
                    (user_id, topic, proficiency_score, questions_attempted, 
                     questions_correct, last_practiced, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (user_id, topic, proficiency, total_questions, correct, now, now))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error updating progress: {e}")
            return False
    
    def get_user_progress(self, user_id: int) -> List[Dict]:
        """Get all progress records for a user"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT topic, subtopic, proficiency_score, 
                       questions_attempted, questions_correct, last_practiced
                FROM learning_progress
                WHERE user_id = ?
                ORDER BY last_practiced DESC
            ''', (user_id,))
            
            rows = cursor.fetchall()
            progress = []
            
            for row in rows:
                progress.append({
                    'topic': row[0],
                    'subtopic': row[1],
                    'proficiency_score': row[2],
                    'questions_attempted': row[3],
                    'questions_correct': row[4],
                    'last_practiced': row[5]
                })
            
            conn.close()
            return progress
        except Exception as e:
            print(f"Error getting user progress: {e}")
            return []
    
    def save_explanation(self, user_id: int, topic: str, explanation: str, tags: List[str] = None):
        """Save a user's explanation for later reference"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO saved_explanations (user_id, topic, explanation, tags, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                user_id,
                topic,
                explanation,
                json.dumps(tags or []),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error saving explanation: {e}")
            return False
    
    def get_saved_explanations(self, user_id: int):
        """Get all saved explanations for a user"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, topic, explanation, tags, created_at
                FROM saved_explanations
                WHERE user_id = ?
                ORDER BY created_at DESC
            ''', (user_id,))
            
            rows = cursor.fetchall()
            explanations = []
            
            for row in rows:
                explanations.append({
                    'id': row[0],
                    'topic': row[1],
                    'explanation': row[2],
                    'tags': json.loads(row[3]) if row[3] else [],
                    'created_at': row[4]
                })
            
            conn.close()
            return explanations
        except Exception as e:
            print(f"Error getting saved explanations: {e}")
            return []
    
    def save_quiz_result(self, user_id: int, topic: str, question: str, 
                        user_answer: str, correct_answer: str, is_correct: bool):
        """Save individual quiz results"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO quiz_results (user_id, topic, question, user_answer, 
                                        correct_answer, is_correct, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                topic,
                question[:500],  # Limit question length
                user_answer[:500],
                correct_answer[:500],
                1 if is_correct else 0,
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            # Update overall progress
            self.update_progress(user_id, topic, 1 if is_correct else 0, 1)
            return True
        except Exception as e:
            print(f"Error saving quiz result: {e}")
            return False
    
    def get_quiz_results(self, user_id: int, topic: str = None):
        """Get quiz results for a user"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            if topic:
                cursor.execute('''
                    SELECT topic, question, user_answer, correct_answer, 
                           is_correct, created_at
                    FROM quiz_results
                    WHERE user_id = ? AND topic = ?
                    ORDER BY created_at DESC
                ''', (user_id, topic))
            else:
                cursor.execute('''
                    SELECT topic, question, user_answer, correct_answer, 
                           is_correct, created_at
                    FROM quiz_results
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                ''', (user_id,))
            
            rows = cursor.fetchall()
            results = []
            
            for row in rows:
                results.append({
                    'topic': row[0],
                    'question': row[1],
                    'user_answer': row[2],
                    'correct_answer': row[3],
                    'is_correct': bool(row[4]),
                    'created_at': row[5]
                })
            
            conn.close()
            return results
        except Exception as e:
            print(f"Error getting quiz results: {e}")
            return []
    
    def add_learning_resource(self, title: str, description: str, url: str, 
                            topic: str, difficulty: str = 'beginner', 
                            content_type: str = 'article', tags: List[str] = None):
        """Add a learning resource to the database"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO learning_resources 
                (title, description, url, topic, difficulty, content_type, tags, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                title,
                description,
                url,
                topic,
                difficulty,
                content_type,
                json.dumps(tags or []),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error adding learning resource: {e}")
            return False
    
    def get_learning_resources(self, topic: str = None, difficulty: str = None):
        """Get learning resources with optional filters"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            query = "SELECT * FROM learning_resources WHERE 1=1"
            params = []
            
            if topic:
                query += " AND topic LIKE ?"
                params.append(f"%{topic}%")
            
            if difficulty and difficulty != 'all':
                query += " AND difficulty = ?"
                params.append(difficulty)
            
            query += " ORDER BY created_at DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            resources = []
            for row in rows:
                resources.append({
                    'id': row[0],
                    'title': row[1],
                    'description': row[2],
                    'url': row[3],
                    'topic': row[4],
                    'difficulty': row[5],
                    'content_type': row[6],
                    'tags': json.loads(row[7]) if row[7] else [],
                    'created_at': row[8]
                })
            
            conn.close()
            return resources
        except Exception as e:
            print(f"Error getting learning resources: {e}")
            return []

# Initialize database
db = DatabaseManager()