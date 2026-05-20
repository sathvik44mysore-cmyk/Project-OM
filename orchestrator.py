import torch
import torch.nn as nn
from phase3_train_model import IntentModel, tokenize, bag_of_words
from phase2_multi_agent import app as research_app
from langchain_core.messages import HumanMessage
import os

# 1. Load the Custom "Brain" (Neural Network)
def load_router():
    FILE = "intent_model.pth"
    if not os.path.exists(FILE):
        raise FileNotFoundError("Model file not found. Please run Phase 3 training first.")
    
    data = torch.load(FILE)
    model_state = data["model_state"]
    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    num_classes = data["num_classes"]
    all_words = data["all_words"]
    tags = data["tags"]

    model = IntentModel(input_size, hidden_size, num_classes)
    model.load_state_dict(model_state)
    model.eval()
    
    return model, all_words, tags

# 2. The Orchestration Logic
class JarvisOrchestrator:
    def __init__(self):
        self.model, self.all_words, self.tags = load_router()
        print("[JARVIS] Neural Network router loaded successfully.")

    def classify_intent(self, text):
        X = torch.FloatTensor([bag_of_words(tokenize(text), self.all_words)])
        output = self.model(X)
        _, predicted = torch.max(output, dim=1)
        return self.tags[predicted.item()]

    def execute(self, user_query):
        print(f"\n[User]: {user_query}")
        
        # Step 1: Route using the Neural Network
        intent = self.classify_intent(user_query)
        print(f"[Orchestrator] Intent Classified as: {intent}")

        # Step 2: Delegate to the correct Department
        if intent == "Research":
            print("[Orchestrator] Routing to Research Department (Multi-Agent)...")
            initial_input = {
                "messages": [HumanMessage(content=user_query)],
                "iterations": 0
            }
            final_state = research_app.invoke(initial_input)
            return final_state['research']
            
        elif intent == "DevOps":
            return "Routing to DevOps Department... (Feature coming in Phase 4.5)"
            
        else:
            return "I'm not sure how to handle that yet, but I can chat! (General Intent)"

# 3. Main JARVIS Loop (CLI Version)
if __name__ == "__main__":
    jarvis = JarvisOrchestrator()
    print("--- JARVIS System Online (Phase 4 Integration) ---")
    
    # Test Queries
    queries = [
        "Research the impact of hydrogen fuel cells in trucking.",
        "Check my repository for security vulnerabilities.",
        "Tell me a story about a brave robot."
    ]
    
    for q in queries:
        response = jarvis.execute(q)
        print(f"[JARVIS Response]: {response[:300]}...") # Truncated
        print("-" * 50)
