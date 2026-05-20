# Project OM: Omni-Agent Learning System (JARVIS)

Project OM is a comprehensive AI engineering project that demonstrates the integration of **Multi-Agent Systems**, the **Model Context Protocol (MCP)**, and **Custom Neural Networks**.

## 🚀 Overview
JARVIS (Just A Rather Very Intelligent System) is a modular AI assistant that can route user queries to specialized departments. It uses a custom-built Neural Network for high-speed intent classification and a multi-agent team (Researcher & Reviewer) for deep information gathering.

## 🛠️ Tech Stack
- **AI Engine:** [Ollama](https://ollama.com/) (Local Llama 3 / Phi-3)
- **Multi-Agent Framework:** [LangGraph](https://www.langchain.com/langgraph)
- **Deep Learning:** [PyTorch](https://pytorch.org/) (Custom Intent Classifier)
- **Web UI:** [Streamlit](https://streamlit.io/)
- **Version Control:** Git

## 📂 Project Structure
- `app.py`: The Streamlit web interface.
- `orchestrator.py`: The central system that routes queries using the Neural Network.
- `phase2_multi_agent.py`: LangGraph implementation for agent collaboration.
- `phase3_train_model.py`: PyTorch training script for the Intent Classifier.
- `intent_model.pth`: The trained weights of your custom neural network.

## ⚙️ Setup & Installation
1. **Prerequisites:** Install [Ollama](https://ollama.com/) and [Git](https://git-scm.com/).
2. **Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Run Ollama:**
   ```bash
   ollama serve
   ollama pull phi3
   ```
4. **Launch JARVIS:**
   ```bash
   streamlit run app.py
   ```

## 🧠 Learning Highlights
- **Under the Hood:** Building a PyTorch model from scratch to understand Deep Learning fundamentals.
- **Orchestration:** Managing agent dialogue and state using LangGraph.
- **System Design:** Building a scalable, modular architecture ready for new tools and agents.

---
*Created by Sathvik Lokesh.*
