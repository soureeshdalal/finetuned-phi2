<h1>Fine-Tuned Phi-2 LoRA Chatbot</h1>

<p>A fine-tuned LoRA adapter for <strong>Microsoft Phi-2</strong>, optimized for <strong>reasoning, STEM, and code generation</strong>. This model is designed to handle structured tasks such as programming, mathematical reasoning, and logical inference.</p>

<h2>Overview</h2>
<p>This project fine-tunes Microsoft's Phi-2 using <strong>LoRA (Low-Rank Adaptation)</strong>, a parameter-efficient fine-tuning method that allows for adapting the model with minimal computational overhead. The LoRA adapter is applied at runtime, keeping the base model frozen while modifying only a small number of trainable parameters. This leads to:</p>
<ul>
  <li><strong>Memory Efficiency</strong> - Only small adapters are updated instead of the full model.</li>
  <li><strong>Faster Training & Inference</strong> - LoRA reduces fine-tuning time and keeps the model lightweight.</li>
  <li><strong>Code and STEM Optimization</strong> - The model retains general knowledge from Phi-2 but enhances performance in coding, math, and structured reasoning.</li>
</ul>

<h2>Model Details</h2>
<ul>
  <li><strong>Base Model:</strong> Microsoft Phi-2</li>
  <li><strong>Fine-Tuning Method:</strong> LoRA (Parameter-Efficient Fine-Tuning)</li>
  <li><strong>Deployment:</strong> Hugging Face Spaces</li>
  <li><strong>Use Cases:</strong> Code generation, STEM tasks, Logical Reasoning, Q&A</li>
</ul>

<h2>Model & Adapter Files</h2>
<table>
  <tr>
    <th>File</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>adapter_config.json</td>
    <td>Configuration for the LoRA adapter</td>
  </tr>
  <tr>
    <td>adapter_model.safetensors</td>
    <td>Fine-tuned LoRA adapter weights</td>
  </tr>
  <tr>
    <td>tokenizer.json</td>
    <td>Tokenizer settings and vocabulary</td>
  </tr>
  <tr>
    <td>special_tokens_map.json</td>
    <td>Special tokens mapping for fine-tuned model</td>
  </tr>
  <tr>
    <td>README.md</td>
    <td>Project documentation</td>
  </tr>
</table>

<h2>Installation & Setup</h2>
<h3>1. Install Dependencies</h3>
<p>Ensure you have the required packages installed:</p>
<pre><code>pip install transformers peft torch gradio huggingface_hub safetensors</code></pre>

<h3>2. Load the Fine-Tuned LoRA Adapter</h3>
<p>Use the following script to apply the fine-tuned LoRA adapter over Microsoft Phi-2:</p>

<pre><code>import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = "microsoft/phi-2"
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="auto")

# Load LoRA adapter
adapter_repo = "soureesh1211/finetuned-phi2"
model = PeftModel.from_pretrained(model, adapter_repo)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Merge LoRA with base model
model = model.merge_and_unload()</code></pre>

<h3>3. Run Inference</h3>
<pre><code>input_text = "Write a Python function to compute the factorial of a number."
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Response:", response)</code></pre>

<h2>🖥️ Running the Chatbot</h2>
<p>A Gradio-based chatbot has been deployed for easy interaction with the fine-tuned model.</p>

<h3>1. Clone the Repository</h3>
<pre><code>git clone https://huggingface.co/spaces/soureesh1211/finetuned-phi2-chatbot
cd finetuned-phi2-chatbot</code></pre>

<h3>2. Install Dependencies</h3>
<pre><code>pip install -r requirements.txt</code></pre>

<h3>3. Run the Chatbot</h3>
<pre><code>python app.py</code></pre>

<h2>Hugging Face Model & Space</h2>
<ul>
  <li><strong>Fine-Tuned Model:</strong> <a href="https://huggingface.co/soureesh1211/finetuned-phi2">soureesh1211/finetuned-phi2</a></li>
  <li><strong>Live Chatbot:</strong> <a href="https://huggingface.co/spaces/soureesh1211/finetuned-phi2-chatbot">Hugging Face Space</a></li>
</ul>
