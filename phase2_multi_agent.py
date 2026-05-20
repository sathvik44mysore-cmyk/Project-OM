import operator
from typing import Annotated, TypedDict, List
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

# 1. Define the State
# This is the "shared memory" between agents.
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    research: str
    critique: str
    iterations: int

# 2. Initialize the Local Brain (Ollama)
llm = ChatOllama(model="phi3")

# 3. Define the Agents (Nodes)

def researcher_node(state: AgentState):
    print("--- [Agent] Researcher is working ---")
    messages = state['messages']
    topic = messages[0].content
    
    # If there is a critique, the researcher should improve the research
    prompt = f"Research the following topic: {topic}."
    if state.get('critique'):
        prompt += f"\n\nEarlier research was critiqued: {state['critique']}. Please improve it."

    response = llm.invoke(prompt)
    return {
        "research": response.content,
        "iterations": state.get('iterations', 0) + 1
    }

def reviewer_node(state: AgentState):
    print("--- [Agent] Reviewer is evaluating ---")
    research = state['research']
    
    prompt = f"Review the following research and provide 2 constructive critiques to improve it:\n\n{research}"
    response = llm.invoke(prompt)
    
    return {
        "critique": response.content
    }

# 4. Define the Graph Logic (The "Orchestrator")

def should_continue(state: AgentState):
    # For this learning phase, we'll stop after 1 iteration to keep it simple.
    if state.get('iterations', 0) >= 1:
        return "end"
    else:
        return "continue"

# 5. Build the Graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("researcher", researcher_node)
workflow.add_node("reviewer", reviewer_node)

# Define edges
workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "reviewer")

# Conditional edge to stop or loop
workflow.add_conditional_edges(
    "reviewer",
    should_continue,
    {
        "continue": "researcher",
        "end": END
    }
)

# Compile the graph
app = workflow.compile()

# 6. Run the Multi-Agent System
if __name__ == "__main__":
    print("--- Project OM: Phase 2 Multi-Agent Collaboration ---")
    
    initial_input = {
        "messages": [HumanMessage(content="Explain the impact of AI on the automotive industry.")],
        "iterations": 0
    }
    
    # Run the graph
    final_state = app.invoke(initial_input)
    
    print("\n--- Final Results ---")
    print("\n[Researcher's Final Output]:")
    print(final_state['research'][:500] + "...") # Truncated for display
    print("\n[Reviewer's Final Critique]:")
    print(final_state['critique'])
