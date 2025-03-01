<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fine-Tuned Phi-2 Model</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }
        h1, h2 { color: #4A90E2; }
        code { background-color: #f4f4f4; padding: 3px; border-radius: 4px; }
        pre { background: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }
    </style>
</head>
<body>
    <h1>💬 Fine-Tuned Phi-2 Model</h1>
    <p>This project fine-tunes <strong>Microsoft's Phi-2</strong> model using <strong>LoRA (Low-Rank Adaptation)</strong> to enhance its performance while maintaining efficiency. The model is deployed using <strong>Gradio</strong> for an interactive chatbot interface.</p>
    
    <h2>🚀 Model Details</h2>
    <ul>
        <li><strong>Base Model:</strong> <code>microsoft/phi-2</code></li>
        <li><strong>Fine-Tuned Model:</strong> <a href="https://huggingface.co/soureesh1211/finetuned-phi2">soureesh1211/finetuned-phi2</a></li>
        <li><strong>Fine-Tuning Technique:</strong> LoRA (Low-Rank Adaptation)</li>
        <li><strong>Frameworks Used:</strong> <code>Hugging Face Transformers</code>, <code>PEFT</code>, <code>Gradio</code></li>
    </ul>
    
    <h2>📌 Features</h2>
    <ul>
        <li>LoRA-based fine-tuning for parameter-efficient training.</li>
        <li>Optimized for conversational AI and chatbot applications.</li>
        <li>Lightweight and efficient deployment using <strong>Gradio</strong>.</li>
        <li>Supports <strong>FP16</strong> for better performance.</li>
    </ul>
    
    <h2>🛠️ Installation</h2>
    <pre><code>pip install -r requirements.txt</code></pre>
    
    <h2>📜 Usage</h2>
    <p>Run the chatbot using:</p>
    <pre><code>python app.py</code></pre>
    
    <h2>📂 Project Files</h2>
    <ul>
        <li><code>app.py</code> - Gradio-based chatbot implementation.</li>
        <li><code>finetune_phi2.ipynb</code> - Fine-tuning script for the model.</li>
        <li><code>requirements.txt</code> - Dependencies required for running the project.</li>
    </ul>
    
    <h2>🌐 Model Deployment</h2>
    <p>The model is deployed on Hugging Face and can be accessed here: <a href="https://huggingface.co/soureesh1211/finetuned-phi2">Phi-2 Fine-Tuned Model</a></p>
    
    <h2>📜 License</h2>
    <p>This project is released under the <strong>Apache-2.0</strong> license.</p>
</body>
</html>
