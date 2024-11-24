from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import torch

# Model name
model_name = "Sidharthan/gemma2_scripter"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

# Determine device (GPU or CPU)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load model
model = AutoPeftModelForCausalLM.from_pretrained(
    model_name,
    device_map=None,  # Manual device placement
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

# Move model to the appropriate device
model = model.to(device)

# Script generation function
def generate_script(prompt):
    
    """Generate a response from the model."""
    # Add the prompt format with special tokens
    formatted_prompt = f"<bos><start_of_turn>keywords\n{prompt}<end_of_turn>\n<start_of_turn>script\n"
    
    # Tokenize input and move tensors to the device
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Generate the response
    outputs = model.generate(
        **inputs,
        max_length=1024,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.2,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Move outputs back to CPU for decoding
    outputs = outputs.cpu()
    
    # Decode and clean the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

# Loop to test the generation multiple times
print("Type 'exit' to terminate the loop.")
while True:
    prompt = input("Enter the keywords: ")
    if prompt.lower() == "exit":
        print("Exiting the loop.")
        break
    response = generate_script(prompt)
    print(f"Generated response: {response}\n")

