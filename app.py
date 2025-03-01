import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load the base model and apply LoRA adapter
base_model = "microsoft/phi-2"  # Base model
adapter_repo = "soureesh1211/finetuned-phi2"  # Your uploaded adapter

# Load the base model
model = AutoModelForCausalLM.from_pretrained(
    base_model, torch_dtype=torch.float16, device_map="auto"
)

# Apply the LoRA adapter
model = PeftModel.from_pretrained(model, adapter_repo)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Define the chatbot function
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

# Set up Gradio UI
iface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=2, placeholder="Enter your prompt..."),
    outputs=gr.Textbox(),
    title="Fine-Tuned Phi-2 LoRA Chatbot",
    description="This chatbot uses a fine-tuned LoRA adapter on Microsoft Phi-2. Enter a prompt and receive a response!"
)

if __name__ == "__main__":
    iface.launch()
