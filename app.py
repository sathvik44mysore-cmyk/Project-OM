import streamlit as st
from orchestrator import JarvisOrchestrator
import time

# 1. Setup Page Configuration
st.set_page_config(page_title="JARVIS: Project OM", page_icon="🤖", layout="wide")

# 2. Initialize the JARVIS Orchestrator (cached to avoid reloading)
@st.cache_resource
def get_jarvis():
    return JarvisOrchestrator()

jarvis = get_jarvis()

# 3. Sidebar Information
with st.sidebar:
    st.title("Project OM")
    st.markdown("""
    **Omni-Agent Learning System**
    - **Brain:** Local Phi-3 (Ollama)
    - **Router:** Custom PyTorch NN
    - **Orchestration:** LangGraph
    """)
    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []

# 4. Chat Interface
st.title("🤖 JARVIS Interface")
st.caption("AI Assistant powered by a Multi-Agent system and Custom Neural Networks.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("How can I help you today?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # JARVIS Logic
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Step 1: Classification Visualization
        intent = jarvis.classify_intent(prompt)
        st.info(f"🧠 **Neural Network Intent:** {intent}")
        
        # Step 2: Agent Collaboration Visualization
        with st.status("Agents are collaborating...", expanded=True) as status:
            st.write("Orchestrator routing query...")
            time.sleep(1)
            if intent == "Research":
                st.write("Department: Research")
                st.write("Agents: Researcher & Reviewer")
            elif intent == "DevOps":
                st.write("Department: DevOps")
            
            # Step 3: Run Execution
            response = jarvis.execute(prompt)
            status.update(label="Response generated!", state="complete", expanded=False)
        
        # Display Final Response
        message_placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.divider()
st.caption("D:\OM Workspace | Powered by Ollama & PyTorch")
