import torch

def inference(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(inputs)
    
    probabilities = torch.softmax(outputs.logits, dim=1)
    sarcasm_probability = probabilities[0][0].item() * 100  # Probability of the first class (sarcastic)
    
    return sarcasm_probability
