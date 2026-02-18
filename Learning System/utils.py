import os
import json
from typing import List, Dict, Any
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
import chromadb
from dotenv import load_dotenv

load_dotenv()

class VectorStoreManager:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        
        # Try to use Gemini embeddings first, fallback to Ollama
        try:
            # Try Google Gemini embeddings
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001"
            )
            print("Using Google Gemini embeddings")
        except Exception as e:
            print(f"Failed to load Gemini embeddings: {e}")
            try:
                # Fallback to Ollama
                self.embeddings = OllamaEmbeddings(model="llama3.2:3b")
                print("Using Ollama embeddings")
            except Exception as e2:
                print(f"Failed to load Ollama embeddings: {e2}")
                raise Exception("No embedding model available. Please set up either Google API or Ollama.")
        
        self.vector_store = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize or load Chroma vector store"""
        try:
            # Check if the directory exists and has files
            if os.path.exists(self.persist_directory) and os.path.isdir(self.persist_directory):
                # List files in the directory
                files = os.listdir(self.persist_directory)
                if files and any(file.endswith('.parquet') or file.endswith('.json') for file in files):
                    # Load existing vector store
                    print("Loading existing vector store...")
                    self.vector_store = Chroma(
                        persist_directory=self.persist_directory,
                        embedding_function=self.embeddings
                    )
                    count = self.vector_store._collection.count()
                    print(f"Loaded existing vector store with {count} documents")
                    if count == 0:
                        self._load_sample_materials()
                else:
                    # Directory exists but is empty, create new with samples
                    print("Creating new vector store with sample materials...")
                    self._create_new_vector_store()
            else:
                # Directory doesn't exist, create new with samples
                print("Creating new vector store with sample materials...")
                os.makedirs(self.persist_directory, exist_ok=True)
                self._create_new_vector_store()
                
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            # Try to create minimal vector store
            try:
                print("Attempting to create minimal vector store...")
                documents = ["Initial document for vector store initialization. This ensures embeddings are generated properly."]
                metadatas = [{
                    'topic': 'Initialization',
                    'subtopic': 'Setup',
                    'difficulty': 'beginner',
                    'content_type': 'explanation',
                    'tags': 'setup,initialization'
                }]
                
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory,
                    metadatas=metadatas,
                    ids=["initial_doc"]
                )
                print("Created minimal vector store for initialization")
                self._load_sample_materials()
            except Exception as e2:
                print(f"Failed to create vector store: {e2}")
                raise
    
    def _create_new_vector_store(self):
        """Create a new vector store with sample materials"""
        sample_materials = self._create_sample_materials()
        documents = []
        metadatas = []
        
        for i, material in enumerate(sample_materials):
            documents.append(material['content'])
            metadatas.append({
                'topic': material['topic'],
                'subtopic': material['subtopic'],
                'difficulty': material['difficulty'],
                'content_type': material['content_type'],
                'tags': ','.join(material['tags'])
            })
        
        # Create the vector store
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            metadatas=metadatas,
            ids=[f"material_{i}" for i in range(len(documents))]
        )
        print(f"Created vector store with {len(documents)} sample materials")
    
    def _load_sample_materials(self):
        """Load sample learning materials if vector store is empty or small"""
        try:
            count = self.vector_store._collection.count()
            if count < 3:  # If we have less than 3 documents, add samples
                print(f"Vector store has only {count} documents, adding sample materials...")
                sample_materials = self._create_sample_materials()
                documents = []
                metadatas = []
                
                for i in range(count, count + len(sample_materials)):
                    material = sample_materials[i - count]
                    documents.append(material['content'])
                    metadatas.append({
                        'topic': material['topic'],
                        'subtopic': material['subtopic'],
                        'difficulty': material['difficulty'],
                        'content_type': material['content_type'],
                        'tags': ','.join(material['tags'])
                    })
                
                if documents:
                    self.vector_store.add_texts(
                        texts=documents,
                        metadatas=metadatas,
                        ids=[f"material_{i}" for i in range(count, count + len(documents))]
                    )
                    print(f"Added {len(documents)} sample materials")
        except Exception as e:
            print(f"Error loading sample materials: {e}")
    
    def _create_sample_materials(self):
        """Create sample learning materials"""
        return [
            {
                'topic': 'Python Programming',
                'subtopic': 'Variables and Data Types',
                'content': '''Variables in Python are like containers that store data values. You don't need to declare variable types explicitly - Python infers it automatically.

Basic data types:
1. Integers: Whole numbers (e.g., 5, -3, 42)
2. Floats: Decimal numbers (e.g., 3.14, -0.001)
3. Strings: Text data (e.g., "Hello", 'Python')
4. Booleans: True or False values
5. Lists: Ordered, mutable collections [1, 2, 3]
6. Dictionaries: Key-value pairs {'name': 'John', 'age': 25}

Example:
x = 10  # integer
name = "Alice"  # string
is_active = True  # boolean''',
                'difficulty': 'beginner',
                'content_type': 'explanation',
                'tags': ['python', 'programming', 'basics']
            },
            {
                'topic': 'Machine Learning',
                'subtopic': 'Introduction to Neural Networks',
                'content': '''Neural networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes (neurons).

Key components:
1. Input Layer: Receives the input data
2. Hidden Layers: Process the data through weighted connections
3. Output Layer: Produces the final prediction

Activation functions (like ReLU or Sigmoid) introduce non-linearity, allowing the network to learn complex patterns.

Training involves:
- Forward pass: Calculate predictions
- Loss calculation: Compare predictions with actual values
- Backward pass: Update weights using gradient descent''',
                'difficulty': 'intermediate',
                'content_type': 'explanation',
                'tags': ['ml', 'ai', 'neural-networks']
            },
            {
                'topic': 'Quantum Computing',
                'subtopic': 'Qubits and Superposition',
                'content': '''Unlike classical bits that are either 0 or 1, qubits can exist in superposition - both 0 and 1 simultaneously.

A qubit state is represented as: |ψ⟩ = α|0⟩ + β|1⟩
where α and β are complex numbers, and |α|² + |β|² = 1

Key concepts:
1. Superposition: Qubit can be in multiple states at once
2. Entanglement: Qubits can be linked, affecting each other instantly
3. Measurement: Collapses superposition to 0 or 1

Example applications:
- Quantum cryptography
- Drug discovery
- Optimization problems''',
                'difficulty': 'advanced',
                'content_type': 'explanation',
                'tags': ['quantum', 'physics', 'computing']
            },
            {
                'topic': 'Web Development',
                'subtopic': 'HTML Basics',
                'content': '''HTML (HyperText Markup Language) is the standard markup language for creating web pages.

Basic structure:
<!DOCTYPE html>
<html>
<head>
    <title>Page Title</title>
</head>
<body>
    <h1>This is a Heading</h1>
    <p>This is a paragraph.</p>
</body>
</html>

Common elements:
- Headings: <h1> to <h6>
- Paragraphs: <p>
- Links: <a href="url">link text</a>
- Images: <img src="image.jpg" alt="description">''',
                'difficulty': 'beginner',
                'content_type': 'tutorial',
                'tags': ['web', 'html', 'frontend']
            },
            {
                'topic': 'Data Science',
                'subtopic': 'Pandas Introduction',
                'content': '''Pandas is a Python library for data manipulation and analysis.

Key data structures:
1. Series: One-dimensional labeled array
2. DataFrame: Two-dimensional labeled data structure (like a spreadsheet)

Basic operations:
import pandas as pd

# Create a DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['NYC', 'LA', 'Chicago']}
df = pd.DataFrame(data)

# Filter data
adults = df[df['Age'] > 30]

# Group by
city_counts = df.groupby('City').size()''',
                'difficulty': 'intermediate',
                'content_type': 'tutorial',
                'tags': ['data-science', 'python', 'pandas']
            }
        ]
    
    def search_materials(self, query: str, filters: Dict = None, k: int = 5):
        """Search for relevant learning materials"""
        try:
            if not self.vector_store:
                self._initialize_vector_store()
            
            if filters:
                filter_dict = {}
                if 'topic' in filters and filters['topic']:
                    filter_dict['topic'] = filters['topic']
                if 'difficulty' in filters and filters['difficulty']:
                    filter_dict['difficulty'] = filters['difficulty']
                if 'content_type' in filters and filters['content_type']:
                    filter_dict['content_type'] = filters['content_type']
                
                if filter_dict:
                    results = self.vector_store.similarity_search(
                        query=query,
                        k=k,
                        filter=filter_dict
                    )
                else:
                    results = self.vector_store.similarity_search(query=query, k=k)
            else:
                results = self.vector_store.similarity_search(query=query, k=k)
            
            return results
        except Exception as e:
            print(f"Error searching materials: {e}")
            # Return empty results instead of crashing
            return []
    
    def add_material(self, content: str, metadata: Dict):
        """Add new learning material to vector store"""
        try:
            self.vector_store.add_texts(
                texts=[content],
                metadatas=[metadata]
            )
            return True
        except Exception as e:
            print(f"Error adding material: {e}")
            return False
    
    def get_all_topics(self):
        """Get all unique topics in the vector store"""
        try:
            collection = self.vector_store._collection
            all_docs = collection.get()
            if all_docs and 'metadatas' in all_docs:
                topics = set()
                for metadata in all_docs['metadatas']:
                    if metadata and 'topic' in metadata:
                        topics.add(metadata['topic'])
                return sorted(list(topics))
        except Exception as e:
            print(f"Error getting topics: {e}")
        return []

class VisualizationUtils:
    @staticmethod
    def create_progress_chart(progress_data):
        """Create progress visualization chart"""
        if not progress_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No progress data yet",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(height=400)
            return fig
        
        df = pd.DataFrame(progress_data)
        # Ensure we have the required columns
        if 'proficiency_score' not in df.columns:
            df['proficiency_score'] = 0
        
        fig = px.bar(
            df,
            x='topic',
            y='proficiency_score',
            color='proficiency_score',
            title='Learning Progress by Topic',
            labels={'proficiency_score': 'Proficiency Score (%)', 'topic': 'Topic'},
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=400,
            hovermode='closest'
        )
        
        return fig
    
    @staticmethod
    def create_activity_timeline(sessions):
        """Create learning activity timeline"""
        if not sessions:
            fig = go.Figure()
            fig.add_annotation(
                text="No learning sessions yet",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(height=400)
            return fig
        
        # Prepare data
        dates = []
        topics = []
        
        for session in sessions:
            if 'created_at' in session and 'topic' in session:
                dates.append(session['created_at'])
                topics.append(session['topic'])
        
        if not dates:
            fig = go.Figure()
            fig.add_annotation(
                text="No valid session data",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            return fig
        
        fig = go.Figure(data=go.Scatter(
            x=dates,
            y=topics,
            mode='markers+lines',
            marker=dict(size=10, color='blue'),
            line=dict(color='lightblue', width=2),
            text=topics,
            hovertemplate='<b>%{y}</b><br>%{x}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Learning Activity Timeline',
            xaxis_title='Date',
            yaxis_title='Topic',
            height=400,
            hovermode='closest'
        )
        
        return fig
    
    @staticmethod
    def create_proficiency_radar(progress_data):
        """Create radar chart for proficiency scores"""
        if len(progress_data) < 3:
            fig = go.Figure()
            fig.add_annotation(
                text="Need at least 3 topics for radar chart",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14)
            )
            fig.update_layout(height=400)
            return fig
        
        topics = [item['topic'] for item in progress_data]
        scores = [item['proficiency_score'] for item in progress_data]
        
        fig = go.Figure(data=go.Scatterpolar(
            r=scores + [scores[0]],  # Close the polygon
            theta=topics + [topics[0]],
            fill='toself',
            line=dict(color='blue', width=2),
            marker=dict(size=8, color='red')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title='Proficiency Radar Chart',
            height=400,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_quiz_accuracy_chart(quiz_results):
        """Create chart showing quiz accuracy over time"""
        if not quiz_results:
            fig = go.Figure()
            fig.add_annotation(
                text="No quiz results yet",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            return fig
        
        df = pd.DataFrame(quiz_results)
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['date'] = df['created_at'].dt.date
        
        # Calculate daily accuracy
        daily_stats = df.groupby('date').agg({
            'is_correct': ['count', 'sum']
        }).reset_index()
        
        daily_stats['accuracy'] = (daily_stats[('is_correct', 'sum')] / 
                                  daily_stats[('is_correct', 'count')]) * 100
        
        fig = go.Figure()
        
        # Add accuracy line
        fig.add_trace(go.Scatter(
            x=daily_stats['date'],
            y=daily_stats['accuracy'],
            mode='lines+markers',
            name='Accuracy',
            line=dict(color='green', width=3),
            marker=dict(size=8)
        ))
        
        # Add question count as bars
        fig.add_trace(go.Bar(
            x=daily_stats['date'],
            y=daily_stats[('is_correct', 'count')],
            name='Questions Attempted',
            yaxis='y2',
            opacity=0.3
        ))
        
        fig.update_layout(
            title='Quiz Performance Over Time',
            xaxis_title='Date',
            yaxis_title='Accuracy (%)',
            yaxis2=dict(
                title='Questions Attempted',
                overlaying='y',
                side='right'
            ),
            height=400,
            hovermode='x unified'
        )
        
        return fig

# Initialize vector store with error handling
try:
    vector_store = VectorStoreManager()
    print("Vector store initialized successfully")
except Exception as e:
    print(f"Failed to initialize vector store: {e}")
    vector_store = None

viz_utils = VisualizationUtils()