<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fine-Tuned Phi-2 LoRA Chatbot</title>
</head>
<body>

<h1 align="center">🚀 Fine-Tuned Phi-2 LoRA Chatbot</h1>
<p align="center">
    A fine-tuned <strong>LoRA adapter</strong> for <strong>Microsoft Phi-2</strong>, optimized for efficiency and performance.<br>
    Lightweight, memory-efficient, and designed for reasoning, STEM, and natural language tasks.
</p>

---

<h2>📌 Overview</h2>
<p>
This project fine-tunes <strong>Microsoft's Phi-2</strong> using <strong>LoRA (Low-Rank Adaptation)</strong>, a parameter-efficient fine-tuning method that allows for adapting the model with minimal computational overhead. The LoRA adapter is applied at runtime, keeping the <strong>base model frozen</strong> while modifying only a small number of trainable parameters. This leads to:
<ul>
    <li>💡 <strong>Memory Efficiency</strong> - Only small adapters are updated instead of the full model.</li>
    <li>⚡ <strong>Faster Training & Inference</strong> - LoRA reduces fine-tuning time and keeps the model lightweight.</li>
    <li>🔍 <strong>Specialized Task Adaptation</strong> - The model retains general knowledge from Phi-2 but enhances performance in target areas.</li>
</ul>
</p>

<h3>🛠 Model Details:</h3>
<ul>
    <li><strong>Base Model:</strong> <a href="https://huggingface.co/microsoft/phi-2">Microsoft Phi-2</a></li>
    <li><strong>Fine-Tuning Method:</strong> LoRA (Parameter-Efficient Fine-Tuning)</li>
    <li><strong>Deployment:</strong> Hugging Face Spaces</li>
    <li><strong>Use Cases:</strong> Reasoning, STEM, Code Understanding, Q&A</li>
</ul>

---

<h2>📂 Model & Adapter Files</h2>
<p>The following files are included in the repository:</p>
<table>
    <tr>
        <th>File</th>
        <th>Description</th>
    </tr>
    <tr>
        <td><code>adapter_config.json</code></td>
        <td>Configuration for the LoRA adapter</td>
    </tr>
    <tr>
        <td><code>adapter_model.safetensors</code></td>
        <td>Fine-tuned LoRA adapter weights</td>
    </tr>
    <tr>
        <td><code>tokenizer.json</code></td>
        <td>Tokenizer settings and vocabulary</td>
    </tr>
    <tr>
        <td><code>special_tokens_map.json</code></td>
        <td>Special tokens mapping for fine-tuned model</td>
    </tr>
    <tr>
        <td><code>README.md</code></td>
        <td>Project documentation</td>
    </tr>
</table>

---

<h2>🛠 Installation & Setup</h2>
<h3>1️⃣ Install Dependencies</h3>
<p>Ensure you have the required packages installed:</p>
<pre>
pip install transformers peft torch gradio huggingface_hub safetensors
</pre>

<h3>2️⃣ Load the Fine-Tuned LoRA Adapter</h3>
<p>Use the following script to apply the fine-tuned LoRA adapter over <strong>Microsoft Phi-2</strong>:</p>

<pre>
import torch
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
model = model.merge_and_unload()
</pre>

<h3>3️⃣ Run Inference</h3>
<pre>
input_text = "What is the capital of France?"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Response:", response)
</pre>

---

<h2>🖥️ Running the Chatbot</h2>
<p>A <strong>Gradio-based chatbot</strong> has been deployed for easy interaction with the fine-tuned model.</p>

<h3>1️⃣ Clone the Repository</h3>
<pre>
git clone https://huggingface.co/spaces/soureesh1211/finetuned-phi2-chatbot
cd finetuned-phi2-chatbot
</pre>

<h3>2️⃣ Install Dependencies</h3>
<pre>
pip install -r requirements.txt
</pre>

<h3>3️⃣ Run the Chatbot</h3>
<pre>
python app.py
</pre>

---

<h2>🌍 Hugging Face Model & Space</h2>
<ul>
    <li><strong>Fine-Tuned Model:</strong> <a href="https://huggingface.co/soureesh1211/finetuned-phi2">soureesh1211/finetuned-phi2</a></li>
    <li><strong>Live Chatbot:</strong> <a href="https://huggingface.co/spaces/soureesh1211/finetuned-phi2-chatbot">Hugging Face Space</a></li>
</ul>

---

<h2>📜 License</h2>
<p>This project is released under the <strong>MIT License</strong>. Feel free to use and modify it.</p>

---

<h2>📢 Acknowledgements</h2>
<ul>
    <li>Microsoft Phi-2: The base model used for fine-tuning.</li>
    <li>Hugging Face: Hosting the model and inference space.</li>
    <li>LoRA (Low-Rank Adaptation): Efficient fine-tuning technique.</li>
</ul>

---

<h2>🎯 Next Steps</h2>
<ul>
    <li>✅ Enhance model capabilities with <strong>RAG (Retrieval-Augmented Generation)</strong></li>
    <li>✅ Fine-tune further for <strong>specific domains</strong> (math, coding, etc.)</li>
    <li>✅ Optimize chatbot <strong>response generation speed</strong></li>
</ul>

<h3 align="center">🚀 Happy Building!</h3>

</body>
</html>
