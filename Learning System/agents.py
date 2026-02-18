import os
import re
from typing import Dict, Any, List, TypedDict
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
import json
from database import db
from utils import vector_store
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class LearningAssistantAgents:
    def __init__(self):
        # Try to use Gemini, fallback to Ollama
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.7
            )
            print("Using Google Gemini LLM")
        except Exception as e:
            print(f"Failed to load Gemini: {e}")
            try:
                self.llm = OllamaLLM(model="llama3.2:3b", temperature=0.7)
                print("Using Ollama LLM")
            except Exception as e2:
                print(f"Failed to load Ollama: {e2}")
                raise Exception("No LLM available. Please set up either Google API or Ollama.")
        
        self.setup_agents()
    
    def setup_agents(self):
        """Setup the three specialized agents"""
        
        # 1. Concept Explainer Agent
        self.concept_explainer_prompt = ChatPromptTemplate.from_template("""
        You are a Concept Explainer AI specializing in breaking down complex topics.
        
        User Profile:
        - Learning Style: {learning_style}
        - Knowledge Level: {knowledge_level}
        - Interests: {interests}
        
        User's Query: {query}
        
        Please provide a clear, structured explanation that:
        1. Starts with a simple analogy or real-world example
        2. Breaks down the concept into manageable parts
        3. Uses appropriate terminology for the user's knowledge level
        4. Includes 1-2 practical examples
        5. Suggests what to learn next
        
        Format your response with:
        - A clear title (## Title)
        - Key points as bullet points
        - Examples in code blocks if programming-related
        - A "Next Steps" section
        
        Make it engaging and easy to understand!
        """)
        
        # 2. Question Generator Agent
        self.question_generator_prompt = ChatPromptTemplate.from_template("""
        You are a Question Generator AI that creates personalized learning questions.
        
        Context:
        - Topic: {topic}
        - Previous Explanation: {explanation}
        - User's Proficiency: {proficiency_level} (0-100)
        
        Generate {num_questions} questions that:
        1. Test understanding of key concepts
        2. Vary in difficulty based on proficiency
        3. Include multiple choice, short answer, and application questions
        4. Provide clear, detailed answers
        
        Format each question as follows:
        
        Q1: [Question text]
        
        Options (if multiple choice):
        A) [Option A]
        B) [Option B]
        C) [Option C]
        D) [Option D]
        
        Answer: [Correct answer]
        
        Explanation: [Detailed explanation of why this is correct]
        
        ---
        
        Keep questions engaging and educational. For short answer questions, provide a model answer.
        """)
        
        # 3. Learning Path Recommender Agent
        self.path_recommender_prompt = ChatPromptTemplate.from_template("""
        You are a Learning Path Recommender AI that creates personalized learning journeys.
        
        User Profile:
        - Current Topic: {current_topic}
        - Knowledge Level: {knowledge_level}
        - Learning Style: {learning_style}
        - Interests: {interests}
        - Progress: {progress_summary}
        
        Available Resources: {available_resources}
        
        Create a personalized learning path that:
        1. Identifies knowledge gaps
        2. Recommends resources in optimal order
        3. Suggests practice exercises
        4. Sets achievable milestones
        5. Estimates time commitment
        
        Format:
        ## Learning Path: [Topic]
        
        ### Current Level Assessment
        [Brief assessment]
        
        ### Recommended Resources (in order):
        1. [Resource Type]: [Title] - [Description]
        
        ### Practice Schedule:
        - Week 1: [Focus area and exercises]
        
        ### Success Metrics:
        - [Metric 1]
        - [Metric 2]
        
        ### Estimated Completion: [Time estimate]
        
        Make it practical and achievable!
        """)
        
        # Create agent chains
        self.concept_explainer_chain = (
            {"query": RunnablePassthrough(), 
             "learning_style": RunnablePassthrough(),
             "knowledge_level": RunnablePassthrough(),
             "interests": RunnablePassthrough()}
            | self.concept_explainer_prompt
            | self.llm
            | StrOutputParser()
        )
        
        self.question_generator_chain = (
            {"topic": RunnablePassthrough(),
             "explanation": RunnablePassthrough(),
             "proficiency_level": RunnablePassthrough(),
             "num_questions": RunnablePassthrough()}
            | self.question_generator_prompt
            | self.llm
            | StrOutputParser()
        )
        
        self.path_recommender_chain = (
            {"current_topic": RunnablePassthrough(),
             "knowledge_level": RunnablePassthrough(),
             "learning_style": RunnablePassthrough(),
             "interests": RunnablePassthrough(),
             "progress_summary": RunnablePassthrough(),
             "available_resources": RunnablePassthrough()}
            | self.path_recommender_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def explain_concept(self, query: str, user_profile: Dict) -> str:
        """Generate explanation for a concept"""
        try:
            print(f"Generating explanation for: {query}")
            explanation = self.concept_explainer_chain.invoke({
                "query": query,
                "learning_style": user_profile.get('learning_style', 'visual'),
                "knowledge_level": user_profile.get('knowledge_level', 'beginner'),
                "interests": ', '.join(user_profile.get('interests', []))
            })
            return explanation
        except Exception as e:
            print(f"Error generating explanation: {str(e)}")
            return f"""## Explanation for: {query}

Sorry, I encountered an error while generating the explanation. 

**Key Points:**
- Please try again with a more specific query
- Make sure your API keys are properly configured
- You can also try rephrasing your question

**Example Query Format:**
- "Explain neural networks for beginners"
- "What is Python programming?"
- "How do quantum computers work?"

Error details: {str(e)[:100]}...
"""
    
    def generate_questions(self, topic: str, explanation: str, 
                          proficiency_level: int, num_questions: int = 3) -> str:
        """Generate practice questions"""
        try:
            print(f"Generating {num_questions} questions for: {topic}")
            questions = self.question_generator_chain.invoke({
                "topic": topic,
                "explanation": explanation[:1000],  # Limit explanation length
                "proficiency_level": proficiency_level,
                "num_questions": num_questions
            })
            return questions
        except Exception as e:
            print(f"Error generating questions: {str(e)}")
            return f"""## Practice Questions for: {topic}

Q1: What are the key concepts you learned about {topic}?

Answer: [Your answer here]
Explanation: This question helps you recall the main points from the explanation.

Q2: Can you provide a real-world example of {topic}?

Answer: [Your example here]
Explanation: Applying concepts to real-world scenarios improves understanding.

Q3: What would you like to learn next about {topic}?

Answer: [Your thoughts here]
Explanation: Identifying next steps helps guide your learning journey.

Note: Error occurred while generating questions: {str(e)[:100]}
"""
    
    def recommend_learning_path(self, current_topic: str, user_profile: Dict, 
                               progress_summary: str, available_resources: List[str]) -> str:
        """Generate personalized learning path"""
        try:
            print(f"Generating learning path for: {current_topic}")
            path = self.path_recommender_chain.invoke({
                "current_topic": current_topic,
                "knowledge_level": user_profile.get('knowledge_level', 'beginner'),
                "learning_style": user_profile.get('learning_style', 'visual'),
                "interests": ', '.join(user_profile.get('interests', [])),
                "progress_summary": progress_summary,
                "available_resources": '\n'.join(available_resources[:5])  # Limit resources
            })
            return path
        except Exception as e:
            print(f"Error generating learning path: {str(e)}")
            return f"""## Learning Path: {current_topic}

### Current Level Assessment
Starting your learning journey on this topic.

### Recommended Learning Approach:
1. **Start with Basics**: Understand fundamental concepts
2. **Practice Regularly**: Apply what you learn through exercises
3. **Build Projects**: Create small projects to reinforce learning
4. **Review and Reflect**: Regularly review what you've learned

### Week-by-Week Plan:
- Week 1: Learn basic concepts and terminology
- Week 2: Practice with simple examples
- Week 3: Build a small project or complete exercises
- Week 4: Review and explore advanced topics

### Success Metrics:
- Complete basic understanding of concepts
- Successfully complete practice exercises
- Build at least one small project

### Estimated Completion: 4 weeks

Note: Error occurred while generating detailed path: {str(e)[:100]}
"""
    
    def parse_questions(self, questions_text: str):
        """Parse generated questions into structured format"""
        questions = []
        current_question = None
        collecting_explanation = False
        explanation_lines = []
        
        lines = questions_text.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines at the start of new question
            if not line and not current_question:
                i += 1
                continue
            
            # Detect new question (Q1:, Question 1:, 1., etc.)
            if (re.match(r'^Q\d+:', line) or 
                re.match(r'^Question \d+:', line) or
                re.match(r'^\d+\.', line)):
                
                # Save previous question if exists
                if current_question:
                    if explanation_lines:
                        current_question['explanation'] = ' '.join(explanation_lines)
                        explanation_lines = []
                    questions.append(current_question)
                    collecting_explanation = False
                
                # Start new question
                current_question = {
                    'question': line,
                    'options': [],
                    'answer': '',
                    'explanation': ''
                }
            
            # Detect options
            elif line and current_question and re.match(r'^[A-D]\)', line):
                current_question['options'].append(line)
            
            # Detect answer
            elif line.startswith('Answer:'):
                if current_question:
                    current_question['answer'] = line.replace('Answer:', '').strip()
            
            # Detect explanation start
            elif line.startswith('Explanation:'):
                collecting_explanation = True
                explanation_lines = [line.replace('Explanation:', '').strip()]
            
            # Collect explanation lines
            elif collecting_explanation and current_question:
                if line:  # Only add non-empty lines
                    explanation_lines.append(line)
            
            i += 1
        
        # Add the last question
        if current_question:
            if explanation_lines:
                current_question['explanation'] = ' '.join(explanation_lines)
            questions.append(current_question)
        
        # If no structured questions found, create a simple one
        if not questions:
            questions.append({
                'question': f'What did you learn about this topic?',
                'options': [],
                'answer': 'Reflect on the key concepts explained above.',
                'explanation': 'This question helps reinforce your learning through reflection.'
            })
        
        return questions
    
    def orchestrate_learning_session(self, query: str, user_id: int, user_profile: Dict):
        """Orchestrate complete learning session using LangGraph"""
        
        # Define state structure
        class LearningState(TypedDict):
            query: str
            user_id: int
            user_profile: Dict
            explanation: str
            questions: str
            learning_path: str
            session_summary: Dict
            
        # Define nodes
        def generate_explanation(state: LearningState) -> LearningState:
            """Node 1: Generate concept explanation"""
            print(f"Node 1: Generating explanation for {state['query']}")
            explanation = self.explain_concept(
                state['query'], 
                state['user_profile']
            )
            return {**state, 'explanation': explanation}
        
        def generate_practice_questions(state: LearningState) -> LearningState:
            """Node 2: Generate practice questions"""
            print(f"Node 2: Generating questions for {state['query']}")
            
            # Get user's proficiency for the topic
            progress = db.get_user_progress(state['user_id'])
            topic_proficiency = 50  # Default
            
            for item in progress:
                if item['topic'].lower() in state['query'].lower():
                    topic_proficiency = item['proficiency_score']
                    break
            
            questions = self.generate_questions(
                topic=state['query'],
                explanation=state['explanation'],
                proficiency_level=topic_proficiency,
                num_questions=3
            )
            return {**state, 'questions': questions}
        
        def generate_learning_path(state: LearningState) -> LearningState:
            """Node 3: Generate learning path"""
            print(f"Node 3: Generating learning path for {state['query']}")
            
            # Search for relevant resources
            resources = []
            if vector_store:
                try:
                    search_results = vector_store.search_materials(
                        state['query'], 
                        k=3
                    )
                    resources = [f"{r.metadata.get('topic', 'Resource')}: {r.page_content[:100]}..." 
                                for r in search_results]
                except Exception as e:
                    print(f"Error searching materials: {e}")
                    resources = ["Sample resources for learning"]
            
            if not resources:
                resources = ["Basic learning materials", "Online tutorials", "Practice exercises"]
            
            progress_summary = "Starting new learning topic"
            
            path = self.recommend_learning_path(
                current_topic=state['query'],
                user_profile=state['user_profile'],
                progress_summary=progress_summary,
                available_resources=resources
            )
            return {**state, 'learning_path': path}
        
        def create_session_summary(state: LearningState) -> LearningState:
            """Node 4: Create session summary"""
            print("Node 4: Creating session summary")
            summary = {
                'query': state['query'],
                'explanation_length': len(state['explanation']),
                'questions_generated': state['questions'].count('Q'),
                'has_learning_path': bool(state['learning_path']),
                'timestamp': str(datetime.now()),
                'user_profile': {
                    'learning_style': state['user_profile'].get('learning_style'),
                    'knowledge_level': state['user_profile'].get('knowledge_level')
                }
            }
            return {**state, 'session_summary': summary}
        
        # Build the graph
        workflow = StateGraph(LearningState)
        
        # Add nodes
        workflow.add_node("explainer", generate_explanation)
        workflow.add_node("question_generator", generate_practice_questions)
        workflow.add_node("path_recommender", generate_learning_path)
        workflow.add_node("summary", create_session_summary)
        
        # Add edges
        workflow.set_entry_point("explainer")
        workflow.add_edge("explainer", "question_generator")
        workflow.add_edge("question_generator", "path_recommender")
        workflow.add_edge("path_recommender", "summary")
        workflow.add_edge("summary", END)
        
        # Compile the graph
        app = workflow.compile()
        
        # Execute the graph
        initial_state = {
            "query": query,
            "user_id": user_id,
            "user_profile": user_profile,
            "explanation": "",
            "questions": "",
            "learning_path": "",
            "session_summary": {}
        }
        
        try:
            result = app.invoke(initial_state)
            
            # Save session to database
            db.save_learning_session(
                user_id=user_id,
                topic=query,
                subtopic="General",
                session_data=result['session_summary']
            )
            
            return result
        except Exception as e:
            print(f"Error in orchestration: {e}")
            # Return a fallback result
            return {
                'explanation': self.explain_concept(query, user_profile),
                'questions': "Error generating questions. Please try again.",
                'learning_path': "Error generating learning path.",
                'session_summary': {'error': str(e), 'query': query}
            }

# Initialize agents
try:
    agents = LearningAssistantAgents()
    print("Agents initialized successfully")
except Exception as e:
    print(f"Failed to initialize agents: {e}")
    agents = None