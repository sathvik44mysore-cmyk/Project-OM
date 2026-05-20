import ollama
import os

# 1. Define our "Tool" logic
def read_local_file(file_path):
    print(f"[Tool] Reading file: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

# 2. Setup the conversation
def run_agent():
    print("--- Project OM: Phase 1 Hello World (Ollama Direct) ---")
    
    model = 'phi3'
    prompt = "Read the README.md file in the current directory and give me a 2-sentence summary of what Project OM is about."
    
    # In a real MCP setup, the LLM would decide to call the tool.
    # For this Hello World, we'll demonstrate the "Hand-off" manually
    # to show how the context flows.
    
    # Step 1: Agent decides it needs to read the file
    print(f"User Query: {prompt}")
    
    # Step 2: Call the tool
    file_content = read_local_file("README.md")
    
    # Step 3: Pass the content to the LLM for summarization
    system_instruction = "You are a helpful assistant. Use the provided file content to answer the user's request."
    context = f"File Content of README.md:\n\n{file_content}"
    
    response = ollama.chat(model=model, messages=[
        {'role': 'system', 'content': system_instruction},
        {'role': 'user', 'content': f"{context}\n\nUser Request: {prompt}"}
    ])
    
    print("\nFinal Agent Response:")
    print(response['message']['content'])

if __name__ == "__main__":
    run_agent()
