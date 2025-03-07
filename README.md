# 💬 Fine-Tuned Phi-2 Model

  <p align="left">
  <a href="https://huggingface.co/spaces/soureesh1211/finetuned-phi2-chatbot">
    <img src="https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue.svg" alt="Open in Spaces">
  </a>
</p>

This project fine-tunes **Microsoft's Phi-2** model using **LoRA (Low-Rank Adaptation)** to enhance its performance while maintaining efficiency. The model is deployed using **Gradio** for an interactive chatbot interface.

## Model Details
- **Base Model:** `microsoft/phi-2`
- **Fine-Tuned Model:** [soureesh1211/finetuned-phi2](https://huggingface.co/soureesh1211/finetuned-phi2)
- **Fine-Tuning Technique:** LoRA (Low-Rank Adaptation)
- **Frameworks Used:** `Hugging Face Transformers`, `PEFT`, `Gradio`

## Features
- LoRA-based fine-tuning for parameter-efficient training.
- Optimized for conversational AI and chatbot applications.
- Lightweight and efficient deployment using **Gradio**.
- Supports **FP16** for better performance.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
Run the chatbot using:
```bash
python app.py
```

## Project Files
- `app.py` - Gradio-based chatbot implementation.
- `finetune_phi2.ipynb` - Fine-tuning script for the model.
- `requirements.txt` - Dependencies required for running the project.

## Model Deployment
The model is deployed on Hugging Face and can be accessed here:([https://huggingface.co/soureesh1211/finetuned-phi2](https://huggingface.co/spaces/soureesh1211/finetuned-phi2-chatbot))

