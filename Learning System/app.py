import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from datetime import datetime
import json
import time

# Import modules with error handling
try:
    from database import db
except ImportError as e:
    st.error(f"Database module import error: {e}")
    db = None

try:
    from agents import agents
except ImportError as e:
    st.error(f"Agents module import error: {e}")
    agents = None

try:
    from utils import vector_store, viz_utils
except ImportError as e:
    st.error(f"Utils module import error: {e}")
    vector_store = None
    viz_utils = None

# Page configuration
st.set_page_config(
    page_title="AI Learning Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state with default values
def init_session_state():
    """Initialize all session state variables with defaults"""
    defaults = {
        'user_id': None,
        'user_profile': None,
        'current_topic': None,
        'current_explanation': None,
        'current_questions': None,
        'current_learning_path': None,
        'parsed_questions': [],
        'quiz_answers': {},
        'quiz_checked': {},
        'show_explanation_saved': False,
        'menu_selection': 'Home'  # Default menu selection
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Initialize session state
init_session_state()

def show_home_page():
    """Home page for non-logged in users"""
    st.title("🧠 AI-Personalized Learning Assistant")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🤖 Smart AI Agents")
        st.info("""
        1. **Concept Explainer** - Breaks down complex topics
        2. **Question Generator** - Creates personalized quizzes
        3. **Path Recommender** - Suggests learning roadmaps
        """)
    
    with col2:
        st.subheader("📚 Personalized Learning")
        st.info("""
        • Adapts to your learning style  
        • Tracks your progress  
        • Identifies knowledge gaps  
        • Recommends resources  
        """)
    
    with col3:
        st.subheader("📊 Progress Analytics")
        st.info("""
        • Visual progress charts  
        • Activity timeline  
        • Proficiency scores  
        • Learning insights  
        """)
    
    st.markdown("---")
    
    # Demo section
    st.subheader("🚀 How It Works")
    
    with st.expander("See a quick demo", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 1. Ask a Question")
            st.code("Explain neural networks like I'm a beginner", language="markdown")
        
        with col2:
            st.write("### 2. Get AI-Powered Help")
            st.code("""
            🤖 Concept Explainer: 
            Neural networks are like digital brains...
            
            ❓ Question Generator:
            Q1: What is a neuron in neural networks?
            
            🗺️ Learning Path Recommender:
            Week 1: Start with linear algebra basics...
            """, language="markdown")
    
    st.markdown("---")
    st.write("### 🎯 Get Started")
    st.markdown("""
    1. **Sign up** for a free account
    2. **Login** to access personalized features
    3. **Start learning** with AI assistance
    4. **Track progress** with visual analytics
    """)

def show_login_page():
    """Login page"""
    st.subheader("🔐 Login")
    
    with st.form("login_form"):
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        submit = st.form_submit_button("Login", use_container_width=True)
        
        if submit:
            if not username or not password:
                st.error("Please fill in all fields")
            elif db is None:
                st.error("Database not available. Please check configuration.")
            else:
                user_id = db.authenticate_user(username, password)
                if user_id:
                    st.session_state.user_id = user_id
                    st.session_state.user_profile = db.get_user_profile(user_id)
                    st.session_state.menu_selection = "Dashboard"  # Reset to Dashboard
                    st.success("Login successful! Redirecting...")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Invalid username or password")

def show_signup_page():
    """Signup page"""
    st.subheader("📝 Sign Up")
    
    with st.form("signup_form"):
        username = st.text_input("Choose a Username", key="signup_username")
        password = st.text_input("Choose a Password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm")
        
        # Learning preferences
        st.write("### Learning Preferences")
        col1, col2 = st.columns(2)
        
        with col1:
            learning_style = st.selectbox(
                "Learning Style",
                ["visual", "auditory", "reading/writing", "kinesthetic"],
                key="signup_style"
            )
        
        with col2:
            knowledge_level = st.selectbox(
                "Current Knowledge Level",
                ["beginner", "intermediate", "advanced"],
                key="signup_level"
            )
        
        interests = st.multiselect(
            "Learning Interests",
            ["programming", "data science", "machine learning", "web development", 
             "mathematics", "physics", "biology", "history", "literature", "business"],
            default=["programming", "data science"],
            key="signup_interests"
        )
        
        submit = st.form_submit_button("Create Account", use_container_width=True)
        
        if submit:
            if not username or not password:
                st.error("Please fill in all required fields")
            elif password != confirm_password:
                st.error("Passwords do not match")
            elif db is None:
                st.error("Database not available. Please check configuration.")
            else:
                success = db.create_user(username, password)
                if success:
                    # Get user_id
                    user_id = db.authenticate_user(username, password)
                    if user_id:
                        # Update profile with preferences
                        db.update_user_profile(
                            user_id=user_id,
                            learning_style=learning_style,
                            knowledge_level=knowledge_level,
                            interests=interests
                        )
                        
                        st.session_state.user_id = user_id
                        st.session_state.user_profile = db.get_user_profile(user_id)
                        st.session_state.menu_selection = "Dashboard"  # Reset to Dashboard
                        st.success("Account created successfully! Redirecting...")
                        time.sleep(1)
                        st.rerun()
                else:
                    st.error("Username already exists")

def show_dashboard():
    """User dashboard after login"""
    st.title(f"Welcome, {st.session_state.user_profile['username']}! 👋")
    
    # Quick stats
    if db:
        progress_data = db.get_user_progress(st.session_state.user_id)
        recent_sessions = db.get_recent_sessions(st.session_state.user_id, limit=3)
    else:
        progress_data = []
        recent_sessions = []
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Topics Learned", len(progress_data))
    
    with col2:
        total_questions = sum(p.get('questions_attempted', 0) for p in progress_data)
        st.metric("Questions Attempted", total_questions)
    
    with col3:
        if progress_data:
            avg_proficiency = sum(p.get('proficiency_score', 0) for p in progress_data) / len(progress_data)
            st.metric("Avg. Proficiency", f"{avg_proficiency:.1f}%")
        else:
            st.metric("Avg. Proficiency", "0%")
    
    with col4:
        st.metric("Learning Sessions", len(recent_sessions))
    
    # Quick actions
    st.markdown("---")
    st.write("### Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎓 Start Learning", use_container_width=True):
            # Set focus to Learn tab
            st.session_state.menu_selection = "Learn"
            st.rerun()
    
    with col2:
        if st.button("📊 View Progress", use_container_width=True):
            st.session_state.menu_selection = "Progress"
            st.rerun()
    
    with col3:
        if st.button("📚 Browse Resources", use_container_width=True):
            st.session_state.menu_selection = "Resources"
            st.rerun()
    
    # Recent activity
    st.markdown("---")
    st.write("### Recent Activity")
    
    if recent_sessions:
        for session in recent_sessions[:3]:
            with st.expander(f"{session.get('topic', 'Unknown')} - {session.get('created_at', '')[:10]}", expanded=False):
                st.write(f"**Topic**: {session.get('topic', 'N/A')}")
                st.write(f"**Date**: {session.get('created_at', 'N/A')}")
                if session.get('session_data'):
                    data = session['session_data']
                    if isinstance(data, dict):
                        st.write(f"**Questions Generated**: {data.get('questions_generated', 0)}")
                        st.write(f"**Explanation Length**: {data.get('explanation_length', 0)} characters")
    else:
        st.info("No recent activity. Start learning to see your progress here!")

def show_learn_page():
    """Main learning interface"""
    st.subheader("🎓 Learn with AI")
    
    # Check if agents are available
    if agents is None:
        st.error("AI Agents are not available. Please check your configuration.")
        st.info("Make sure you have set up either Google API key or Ollama.")
        return
    
    # Topic input
    col1, col2 = st.columns([3, 1])
    with col1:
        topic = st.text_input(
            "What would you like to learn about?",
            placeholder="e.g., Explain neural networks, How does Python work, What is machine learning?",
            key="learn_topic_input"
        )
    with col2:
        num_questions = st.number_input("Questions", min_value=1, max_value=10, value=3, key="num_questions_input")
    
    if st.button("🚀 Start Learning Session", type="primary", use_container_width=True, key="start_learning_btn"):
        if topic:
            st.session_state.current_topic = topic
            
            with st.spinner("🤖 AI agents are preparing your personalized learning session..."):
                # Reset session state for new session
                st.session_state.current_explanation = None
                st.session_state.current_questions = None
                st.session_state.current_learning_path = None
                st.session_state.parsed_questions = []
                st.session_state.quiz_answers = {}
                st.session_state.quiz_checked = {}
                st.session_state.show_explanation_saved = False
                
                # Orchestrate complete learning session
                result = agents.orchestrate_learning_session(
                    query=topic,
                    user_id=st.session_state.user_id,
                    user_profile=st.session_state.user_profile
                )
                
                # Store in session state
                st.session_state.current_explanation = result.get('explanation', '')
                st.session_state.current_questions = result.get('questions', '')
                st.session_state.current_learning_path = result.get('learning_path', '')
                
                # Parse questions
                if st.session_state.current_questions:
                    st.session_state.parsed_questions = agents.parse_questions(st.session_state.current_questions)
                
                # Force rerun to show results
                st.rerun()
        else:
            st.warning("Please enter a topic to learn about")
    
    # Display results if available
    if st.session_state.current_explanation:
        st.markdown("---")
        
        # Display results in tabs
        tab1, tab2, tab3 = st.tabs(["📖 Explanation", "❓ Practice Questions", "🗺️ Learning Path"])
        
        with tab1:
            st.write(f"### 📚 {st.session_state.current_topic}")
            st.markdown(st.session_state.current_explanation)
            
            # Save explanation section
            st.markdown("---")
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("💾 Save Explanation", key="save_explanation_btn", use_container_width=True):
                    if db:
                        tags = st.session_state.user_profile.get('interests', [])
                        if db.save_explanation(
                            user_id=st.session_state.user_id,
                            topic=st.session_state.current_topic,
                            explanation=st.session_state.current_explanation,
                            tags=tags
                        ):
                            st.session_state.show_explanation_saved = True
                            st.success("✅ Explanation saved to your learning history!")
                            # Refresh to show in profile
                            time.sleep(1)
                        else:
                            st.error("Failed to save explanation")
                    else:
                        st.error("Database not available")
            
            with col2:
                if st.session_state.show_explanation_saved:
                    st.success("Explanation saved! View it in your Profile page.")
                else:
                    st.caption("Save this explanation to review later in your Profile")
        
        with tab2:
            st.write("### ❓ Practice Questions")
            
            if st.session_state.current_questions:
                st.markdown(st.session_state.current_questions)
            else:
                st.info("No questions generated for this topic.")
            
            # Interactive quiz section
            if st.session_state.parsed_questions:
                st.markdown("---")
                st.write("### 📝 Interactive Quiz")
                
                for idx, question_data in enumerate(st.session_state.parsed_questions):
                    st.markdown(f"**Q{idx + 1}: {question_data['question']}**")
                    
                    # Display options if available
                    if question_data['options']:
                        for option in question_data['options']:
                            st.write(f"  {option}")
                    
                    # Answer input
                    answer_key = f"quiz_answer_{idx}"
                    if answer_key not in st.session_state.quiz_answers:
                        st.session_state.quiz_answers[answer_key] = ""
                    
                    user_answer = st.text_input(
                        f"Your answer for Question {idx + 1}:",
                        value=st.session_state.quiz_answers[answer_key],
                        key=answer_key,
                        placeholder="Type your answer here..."
                    )
                    st.session_state.quiz_answers[answer_key] = user_answer
                    
                    # Check answer button
                    check_key = f"check_answer_{idx}"
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        if st.button(f"Check Answer {idx + 1}", key=check_key, use_container_width=True):
                            if not user_answer:
                                st.warning("Please enter an answer before checking")
                            elif db:
                                # Simple string comparison (could be enhanced)
                                user_answer_lower = user_answer.lower()
                                correct_answer_lower = question_data['answer'].lower()
                                
                                # Basic similarity check
                                is_correct = (user_answer_lower in correct_answer_lower or 
                                             correct_answer_lower in user_answer_lower or
                                             any(word in user_answer_lower for word in correct_answer_lower.split() if len(word) > 3))
                                
                                # Save result
                                db.save_quiz_result(
                                    user_id=st.session_state.user_id,
                                    topic=st.session_state.current_topic,
                                    question=question_data['question'][:200],
                                    user_answer=user_answer,
                                    correct_answer=question_data['answer'],
                                    is_correct=is_correct
                                )
                                
                                st.session_state.quiz_checked[check_key] = True
                                
                                if is_correct:
                                    st.success("✅ Correct!")
                                else:
                                    st.error(f"❌ The correct answer is: {question_data['answer']}")
                                
                                # Show explanation
                                if question_data.get('explanation'):
                                    st.info(f"**Explanation**: {question_data['explanation']}")
                            else:
                                st.error("Database not available")
                    
                    with col2:
                        if check_key in st.session_state.quiz_checked:
                            st.info("✓ Answer checked")
                    
                    st.markdown("---")
                
                # Quiz summary
                total_questions = len(st.session_state.parsed_questions)
                checked_count = len([k for k in st.session_state.quiz_checked.keys() if 'check_answer_' in k])
                
                if checked_count > 0 and db:
                    # Get recent quiz results for this topic
                    recent_results = db.get_quiz_results(
                        st.session_state.user_id, 
                        st.session_state.current_topic
                    )
                    correct_count = len([r for r in recent_results if r.get('is_correct', False)])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Questions Completed", f"{checked_count}/{total_questions}")
                    with col2:
                        st.metric("Correct Answers", f"{correct_count}/{checked_count}")
                    
                    if checked_count == total_questions:
                        st.balloons()
                        st.success("🎉 Congratulations! You've completed all questions!")
            else:
                st.info("No interactive quiz available for this session.")
        
        with tab3:
            st.write("### 🗺️ Recommended Learning Path")
            
            if st.session_state.current_learning_path:
                st.markdown(st.session_state.current_learning_path)
            else:
                st.info("No learning path generated for this topic.")
            
            # Resource recommendations
            if vector_store:
                st.markdown("---")
                st.write("### 📚 Recommended Resources")
                
                # Search vector store for relevant materials
                search_results = vector_store.search_materials(
                    st.session_state.current_topic, 
                    k=3
                )
                
                if search_results:
                    for i, result in enumerate(search_results):
                        with st.expander(f"Resource {i+1}: {result.metadata.get('topic', 'Topic')}"):
                            st.write(f"**Difficulty**: {result.metadata.get('difficulty', 'N/A')}")
                            st.write(f"**Content Type**: {result.metadata.get('content_type', 'N/A')}")
                            st.write(f"**Content Preview**:")
                            st.write(result.page_content[:200] + "...")
                            
                            if st.button(f"Study This Resource", key=f"study_res_{i}"):
                                # Set as new learning topic
                                st.session_state.current_topic = result.metadata.get('topic', st.session_state.current_topic)
                                st.info(f"Now learning about: {result.metadata.get('topic', 'this topic')}")
                                # Clear current results to trigger new session
                                st.session_state.current_explanation = None
                                st.rerun()
                else:
                    st.info("No additional resources found for this topic.")
            else:
                st.info("Resource search is not available at the moment.")

def show_progress_page():
    """Progress tracking and analytics"""
    st.subheader("📊 Learning Progress")
    
    if db is None:
        st.error("Database not available. Cannot load progress data.")
        return
    
    progress_data = db.get_user_progress(st.session_state.user_id)
    recent_sessions = db.get_recent_sessions(st.session_state.user_id, limit=10)
    
    if progress_data:
        # Convert to DataFrame for display
        df = pd.DataFrame(progress_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Progress Overview")
            # Select and rename columns for better display
            display_df = df.copy()
            if 'last_practiced' in display_df.columns:
                display_df['Last Practice'] = pd.to_datetime(display_df['last_practiced']).dt.strftime('%Y-%m-%d')
            
            # Reorder columns for better display
            display_columns = []
            for col in ['topic', 'proficiency_score', 'questions_attempted', 'questions_correct', 'Last Practice']:
                if col in display_df.columns:
                    display_columns.append(col)
            
            st.dataframe(
                display_df[display_columns].rename(columns={
                    'topic': 'Topic',
                    'proficiency_score': 'Proficiency (%)',
                    'questions_attempted': 'Attempted',
                    'questions_correct': 'Correct'
                }),
                use_container_width=True,
                height=300
            )
        
        with col2:
            st.write("### Recent Sessions")
            if recent_sessions:
                for session in recent_sessions[:3]:
                    with st.expander(f"{session['topic']} - {session['created_at'][:10]}", expanded=False):
                        st.write(f"**Topic**: {session['topic']}")
                        st.write(f"**Date**: {session['created_at'][:19]}")
                        if session['session_data']:
                            data = session['session_data']
                            if isinstance(data, dict):
                                st.write(f"**Questions**: {data.get('questions_generated', 0)}")
                                st.write(f"**Explanation**: {data.get('explanation_length', 0)} chars")
            else:
                st.info("No recent sessions")
        
        # Visualizations
        if viz_utils:
            st.markdown("---")
            st.write("### 📈 Progress Analytics")
            
            tab1, tab2, tab3 = st.tabs(["Progress Chart", "Activity Timeline", "Radar Chart"])
            
            with tab1:
                fig = viz_utils.create_progress_chart(progress_data)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                fig = viz_utils.create_activity_timeline(recent_sessions)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                fig = viz_utils.create_proficiency_radar(progress_data)
                st.plotly_chart(fig, use_container_width=True)
        
        # Quiz results visualization
        quiz_results = db.get_quiz_results(st.session_state.user_id)
        if quiz_results and viz_utils:
            st.markdown("---")
            st.write("### 🎯 Quiz Performance")
            
            fig = viz_utils.create_quiz_accuracy_chart(quiz_results)
            st.plotly_chart(fig, use_container_width=True)
        
        # Export progress
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Export Progress Data", use_container_width=True):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"learning_progress_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col2:
            if st.button("🔄 Reset Progress Data", use_container_width=True):
                st.warning("This will clear all your progress data. Are you sure?")
                confirm = st.checkbox("Yes, I want to reset all my progress")
                if confirm and st.button("Confirm Reset", type="secondary"):
                    st.info("Reset functionality would be implemented here")
                    st.info("Note: In a full implementation, this would delete all progress records")
    
    else:
        st.info("No progress data yet. Start learning to track your progress!")
        st.markdown("""
        Your progress dashboard will show:
        1. **Proficiency scores** for each topic
        2. **Questions attempted** and correct answers
        3. **Learning activity** timeline
        4. **Visual charts** of your progress
        
        Start learning in the **Learn** tab to begin tracking!
        """)

def show_resources_page():
    """Browse learning resources"""
    st.subheader("📚 Learning Resources")
    
    if vector_store is None:
        st.error("Resource search is not available. Vector store not initialized.")
        st.info("Make sure your embedding model is properly configured.")
        return
    
    # Search resources
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input(
            "Search learning resources", 
            placeholder="e.g., python, machine learning, algorithms",
            key="resource_search"
        )
    
    with col2:
        difficulty = st.selectbox(
            "Difficulty", 
            ["all", "beginner", "intermediate", "advanced"],
            key="resource_difficulty"
        )
    
    if st.button("🔍 Search Resources", use_container_width=True, key="search_resources_btn"):
        if search_query:
            filters = {}
            if difficulty != "all":
                filters['difficulty'] = difficulty
            
            with st.spinner("Searching resources..."):
                results = vector_store.search_materials(search_query, filters=filters, k=10)
                
                st.write(f"### Found {len(results)} resources")
                
                for i, result in enumerate(results):
                    with st.expander(f"📄 {result.metadata.get('topic', 'Unknown')}", expanded=False):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write(f"**Topic**: {result.metadata.get('topic', 'N/A')}")
                            st.write(f"**Subtopic**: {result.metadata.get('subtopic', 'N/A')}")
                            st.write(f"**Content Type**: {result.metadata.get('content_type', 'N/A')}")
                            st.write(f"**Difficulty**: {result.metadata.get('difficulty', 'N/A')}")
                            st.write(f"**Tags**: {result.metadata.get('tags', 'N/A')}")
                        
                        with col2:
                            if st.button("Study", key=f"study_res_{i}", use_container_width=True):
                                st.session_state.current_topic = result.metadata.get('topic', search_query)
                                st.session_state.menu_selection = "Learn"
                                st.success(f"Redirecting to learn about: {result.metadata.get('topic', 'this topic')}")
                                time.sleep(1)
                                st.rerun()
                        
                        st.markdown("---")
                        st.write("**Content Preview**:")
                        content_preview = result.page_content
                        if len(content_preview) > 500:
                            content_preview = content_preview[:500] + "..."
                        st.write(content_preview)
                        
                        # Action buttons
                        col1, col2, col3 = st.columns(3)
                        if agents:
                            with col1:
                                if st.button("Explain This", key=f"explain_{i}", use_container_width=True):
                                    with st.spinner("Generating explanation..."):
                                        explanation = agents.explain_concept(
                                            query=result.metadata.get('topic', search_query),
                                            user_profile=st.session_state.user_profile
                                        )
                                        st.write("### 🤖 AI Explanation:")
                                        st.markdown(explanation)
                            
                            with col2:
                                if st.button("Generate Questions", key=f"questions_{i}", use_container_width=True):
                                    with st.spinner("Generating questions..."):
                                        questions = agents.generate_questions(
                                            topic=result.metadata.get('topic', search_query),
                                            explanation=result.page_content[:200],
                                            proficiency_level=50,
                                            num_questions=2
                                        )
                                        st.write("### ❓ Practice Questions:")
                                        st.markdown(questions)
                        
                        with col3:
                            if st.button("Save Resource", key=f"save_res_{i}", use_container_width=True):
                                st.info("Resource saved to your learning list")
        else:
            st.warning("Please enter a search query")
    
    # Show all available topics
    st.markdown("---")
    st.write("### 📂 Available Topics")
    
    topics = vector_store.get_all_topics()
    if topics:
        cols = st.columns(3)
        for i, topic in enumerate(topics):
            with cols[i % 3]:
                if st.button(f"📚 {topic}", use_container_width=True, key=f"topic_{i}"):
                    st.session_state.current_topic = topic
                    st.session_state.menu_selection = "Learn"
                    st.rerun()
    else:
        st.info("No topics available in the resource database.")

def show_profile_page():
    """User profile management"""
    if db is None:
        st.error("Database not available. Cannot load profile data.")
        return
    
    st.subheader("👤 Your Profile")
    
    profile = st.session_state.user_profile
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Current Settings")
        st.info(f"**Username**: {profile['username']}")
        st.info(f"**Learning Style**: {profile['learning_style']}")
        st.info(f"**Knowledge Level**: {profile['knowledge_level']}")
        st.info(f"**Interests**: {', '.join(profile['interests'])}")
    
    with col2:
        st.write("### Update Preferences")
        
        with st.form("update_profile"):
            new_learning_style = st.selectbox(
                "Learning Style",
                ["visual", "auditory", "reading/writing", "kinesthetic"],
                index=["visual", "auditory", "reading/writing", "kinesthetic"].index(
                    profile.get('learning_style', 'visual')
                ),
                key="update_style"
            )
            
            new_knowledge_level = st.selectbox(
                "Knowledge Level",
                ["beginner", "intermediate", "advanced"],
                index=["beginner", "intermediate", "advanced"].index(
                    profile.get('knowledge_level', 'beginner')
                ),
                key="update_level"
            )
            
            new_interests = st.multiselect(
                "Learning Interests",
                ["programming", "data science", "machine learning", "web development", 
                 "mathematics", "physics", "biology", "history", "literature", "business"],
                default=profile.get('interests', []),
                key="update_interests"
            )
            
            if st.form_submit_button("Update Profile", use_container_width=True):
                db.update_user_profile(
                    user_id=st.session_state.user_id,
                    learning_style=new_learning_style,
                    knowledge_level=new_knowledge_level,
                    interests=new_interests
                )
                st.session_state.user_profile = db.get_user_profile(st.session_state.user_id)
                st.success("✅ Profile updated successfully!")
                time.sleep(1)
                st.rerun()
    
    # Saved Explanations Section
    st.markdown("---")
    st.write("### 📚 Saved Explanations")
    
    saved_explanations = db.get_saved_explanations(st.session_state.user_id)
    
    if saved_explanations:
        for explanation in saved_explanations:
            with st.expander(f"{explanation['topic']} - {explanation['created_at'][:10]}", expanded=False):
                st.write(f"**Topic**: {explanation['topic']}")
                st.write(f"**Saved on**: {explanation['created_at'][:19]}")
                st.write(f"**Tags**: {', '.join(explanation['tags'])}")
                
                # Show preview of explanation
                preview = explanation['explanation']
                if len(preview) > 300:
                    preview = preview[:300] + "..."
                st.markdown("**Preview**:")
                st.markdown(preview)
                
                # Options
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"View Full", key=f"view_exp_{explanation['id']}", use_container_width=True):
                        st.session_state.current_topic = explanation['topic']
                        st.session_state.current_explanation = explanation['explanation']
                        st.session_state.menu_selection = "Learn"
                        st.rerun()
                
                with col2:
                    if st.button(f"Delete", key=f"delete_exp_{explanation['id']}", use_container_width=True):
                        st.warning("Delete functionality would be implemented here")
    else:
        st.info("No saved explanations yet. Save explanations from the Learn page!")
    
    # Quiz Results Section
    st.markdown("---")
    st.write("### 📊 Quiz Results History")
    
    quiz_results = db.get_quiz_results(st.session_state.user_id)
    
    if quiz_results:
        # Group by topic
        results_by_topic = {}
        for result in quiz_results:
            topic = result.get('topic', 'Unknown')
            if topic not in results_by_topic:
                results_by_topic[topic] = []
            results_by_topic[topic].append(result)
        
        for topic, results in results_by_topic.items():
            with st.expander(f"{topic} ({len(results)} attempts)", expanded=False):
                correct_count = len([r for r in results if r.get('is_correct', False)])
                accuracy = (correct_count / len(results)) * 100 if results else 0
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accuracy", f"{accuracy:.1f}%")
                with col2:
                    st.metric("Total Attempts", len(results))
                
                for i, result in enumerate(results[-3:]):  # Show last 3 attempts
                    st.write(f"**Q**: {result.get('question', '')[:80]}...")
                    st.write(f"**Your Answer**: {result.get('user_answer', '')[:50]}...")
                    st.write(f"**Correct Answer**: {result.get('correct_answer', '')[:50]}...")
                    st.write(f"**Result**: {'✅ Correct' if result.get('is_correct', False) else '❌ Incorrect'}")
                    st.write(f"**Date**: {result.get('created_at', '')[:19]}")
                    st.markdown("---")
    else:
        st.info("No quiz results yet. Complete some quizzes in the Learn page!")
    
    st.markdown("---")
    st.write("### ⚙️ Account Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📋 View Learning History", use_container_width=True, key="view_history"):
            sessions = db.get_recent_sessions(st.session_state.user_id, limit=20)
            if sessions:
                st.write("### Learning History")
                for session in sessions:
                    with st.expander(f"{session['topic']} - {session['created_at'][:10]}", expanded=False):
                        st.write(f"**Session ID**: {session['id']}")
                        st.write(f"**Topic**: {session['topic']}")
                        st.write(f"**Subtopic**: {session['subtopic']}")
                        st.write(f"**Date**: {session['created_at']}")
                        if session['session_data']:
                            st.json(session['session_data'], expanded=False)
            else:
                st.info("No learning history yet")
    
    with col2:
        if st.button("🚪 Logout", type="secondary", use_container_width=True, key="logout_btn"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            # Reinitialize with defaults
            init_session_state()
            st.success("Logged out successfully!")
            time.sleep(1)
            st.rerun()

# Main application logic
def main():
    # Get the current menu selection safely
    current_selection = st.session_state.menu_selection
    
    # Ensure menu selection is valid for current state
    if st.session_state.user_id is None:
        # Guest menu options
        menu_options = ["Home", "Login", "Sign Up"]
        # If current selection is not in guest options, reset to Home
        if current_selection not in menu_options:
            st.session_state.menu_selection = "Home"
            current_selection = "Home"
    else:
        # User menu options
        menu_options = ["Dashboard", "Learn", "Progress", "Resources", "Profile"]
        # If current selection is not in user options, reset to Dashboard
        if current_selection not in menu_options:
            st.session_state.menu_selection = "Dashboard"
            current_selection = "Dashboard"
    
    # Display horizontal menu based on login status
    if st.session_state.user_id is None:
        # Non-logged in user menu
        selected = option_menu(
            menu_title=None,
            options=menu_options,
            icons=["house", "box-arrow-in-right", "person-plus"],
            menu_icon="cast",
            default_index=menu_options.index(current_selection),
            orientation="horizontal",
            key="guest_menu"
        )
        
        # Update session state
        st.session_state.menu_selection = selected
        
        if selected == "Home":
            show_home_page()
        elif selected == "Login":
            show_login_page()
        elif selected == "Sign Up":
            show_signup_page()
    
    else:
        # Logged in user menu
        selected = option_menu(
            menu_title=None,
            options=menu_options,
            icons=["speedometer", "book", "bar-chart", "folder", "person"],
            menu_icon="cast",
            default_index=menu_options.index(current_selection),
            orientation="horizontal",
            key="user_menu"
        )
        
        # Update session state
        st.session_state.menu_selection = selected
        
        if selected == "Dashboard":
            show_dashboard()
        
        elif selected == "Learn":
            show_learn_page()
        
        elif selected == "Progress":
            show_progress_page()
        
        elif selected == "Resources":
            show_resources_page()
        
        elif selected == "Profile":
            show_profile_page()

if __name__ == "__main__":
    main()