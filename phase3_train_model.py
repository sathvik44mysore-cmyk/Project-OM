import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os

# 1. Prepare Synthetic Dataset
# In a real project, you'd use a large open dataset. 
# Here, we create a small one to understand the workflow.
data = [
    # 0: Research
    ("What are the latest AI trends?", 0),
    ("Search for news about electric cars.", 0),
    ("Analyze the automotive industry growth.", 0),
    ("Tell me about the new LLM models.", 0),
    ("Research the impact of hydrogen fuel cells.", 0),
    ("Find trends in renewable energy.", 0),
    ("What is the current state of quantum computing?", 0),
    
    # 1: DevOps
    ("Check my code for bugs.", 1),
    ("Review this github repository.", 1),
    ("Fix the error in my python script.", 1),
    ("How do I deploy this docker container?", 1),
    ("Scan my repo for vulnerabilities.", 1),
    ("Optimize my database queries.", 1),
    ("Set up a CI/CD pipeline.", 1),
    
    # 2: General
    ("Hello JARVIS, how are you?", 2),
    ("Tell me a joke.", 2),
    ("What time is it?", 2),
    ("Who created you?", 2),
    ("Tell me a story.", 2),
    ("What is your favorite color?", 2)
]

# Simple Bag-of-Words preprocessing
def tokenize(text):
    return text.lower().split()

def bag_of_words(tokenized_sentence, all_words):
    bag = [0.0 for _ in range(len(all_words))]
    for word in tokenized_sentence:
        if word in all_words:
            bag[all_words.index(word)] = 1.0
    return bag

all_words = sorted(list(set([w for s, l in data for w in tokenize(s)])))
tags = ["Research", "DevOps", "General"]

X_train = [bag_of_words(tokenize(s), all_words) for s, l in data]
y_train = [l for s, l in data]

class IntentDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = torch.FloatTensor(X_train)
        self.y_data = torch.LongTensor(y_train)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# 2. Define the Neural Network Architecture
class IntentModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(IntentModel, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # No activation at the end because we use CrossEntropyLoss
        return out

# 3. Training Loop
if __name__ == "__main__":
    print("--- Project OM: Phase 3 Training Custom Neural Network ---")
    
    # Hyperparameters
    input_size = len(all_words)
    hidden_size = 8
    num_classes = len(tags)
    learning_rate = 0.01
    num_epochs = 100
    
    dataset = IntentDataset()
    train_loader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)
    
    device = torch.device('cpu') # Running on CPU for compatibility
    model = IntentModel(input_size, hidden_size, num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(words)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("\nTraining complete!")

    # 4. Save the model and metadata
    model_data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_classes": num_classes,
        "all_words": all_words,
        "tags": tags
    }
    
    FILE = "intent_model.pth"
    torch.save(model_data, FILE)
    print(f"Model saved to {FILE}")

    # 5. Simple Test
    print("\n--- Testing Model Prediction ---")
    test_sentence = "Show me the news about artificial intelligence"
    X = torch.FloatTensor([bag_of_words(tokenize(test_sentence), all_words)])
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    print(f"Input: '{test_sentence}'")
    print(f"Predicted Intent: {tag}")
