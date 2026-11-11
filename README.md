Fine-Tuning SmolLM2-135M Using GRPO

🚀 Fine-tuning a lightweight Large Language Model using Group Relative Policy Optimization (GRPO), LoRA, and deploying it with FastAPI & Streamlit

📌 Project Overview

This project demonstrates fine-tuning the SmolLM2-135M-Instruct model using GRPO (Group Relative Policy Optimization) from the trl library, combined with LoRA (Low-Rank Adaptation) for parameter-efficient training.

The fine-tuned model is deployed as:

FastAPI backend for inference

Streamlit web application for interactive usage

Docker container for easy, all-in-one deployment (Streamlit + FastAPI)

The project focuses on length-controlled text generation, where the reward function encourages outputs close to a target length.

🔗 Live Demo (Streamlit App):

👉 https://7beshoyarnest-finetuning-smollm2-135m-u-streamlit-appapp-3zmsha.streamlit.app/

🧠 Key Concepts Used

GRPO (Group Relative Policy Optimization)

A reinforcement learning method that compares multiple generated outputs per prompt and optimizes based on relative rewards.

LoRA (Low-Rank Adaptation)

Efficient fine-tuning by training only ~3.5% of the model parameters.

TRL (Transformer Reinforcement Learning)

Used to implement GRPO training.

FastAPI

Production-ready inference API.

Streamlit

Lightweight frontend for real-time interaction.

Weights & Biases (W&B)

Experiment tracking and training visualization.

🏗️ Project Structure

FineTuning_SmolLM2-135M_UsingGRPO/

│

├── grpo_finetune.ipynb        # Main training notebook (GRPO + LoRA)

│

├── api/

│   └── main.py                # FastAPI inference backend

│

├── streamlit_app/

│   └── app.py                 # Streamlit frontend application

│

├── training_metrics/          # Plots of reward curves & training stats

│   ├── train_rewards_reward_len_mean.png

│   ├── train_rewards_reward_len_std.png

│   └── train_step_time.png

├── Dockerfile

├── .dockerignore

├── requirements.txt

├── requirements-docker.txt

└── README.md

📊 Dataset

Dataset: mlabonne/smoltldr

Structure: Prompt → Short Summary

Splits:

Train: 2000 samples

Validation: 200 samples

Test: 200 samples

This dataset is designed for length-controlled summarization, making it ideal for GRPO experiments.

Source:

https://huggingface.co/datasets/mlabonne/smoltldr

🤖 Model

Base Model: HuggingFaceTB/SmolLM2-135M-Instruct

Architecture: Causal Language Model

Fine-Tuning Method: LoRA + GRPO

Trainable Parameters: ~4.88M (≈ 3.5%)

Model source:

https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct

⚙️ Training Details

LoRA Configuration

r = 16

lora_alpha = 32

target_modules = "all-linear"

Task type: CAUSAL_LM

Reward Function

The reward function encourages generations close to 50 tokens:

def reward_len(completions, **kwargs):

    return [-abs(50 - len(completion)) for completion in completions]

GRPO Training Configuration

Learning rate: 2e-5

Batch size: 8

Gradient accumulation: 2

Max prompt length: 512

Max completion length: 96

Number of generations per prompt: 8

Epochs: 1

Mixed precision: bf16

Optimizer: AdamW

Logging: Weights & Biases

📈 Experiment Tracking

Training runs are logged using Weights & Biases, including:

Reward distributions

Loss curves

Generation statistics

GRPO profiling metrics

W&B Project:

https://wandb.ai/beshoyarnest01-minia-university/GRPO


## Benchmark Results

Evaluated using [TPN Bench](https://bench.trueperformancenetwork.com/) and [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

| Benchmark | Metric | Score |

|---|---|---:|

| ARC-Challenge | Normalized Accuracy | 27.99% |

**Evaluation Details:**

- Test samples: 1,172
  
- Model revision: `9b01868`
  
- Evaluation mode: Plain prompts
  
- Runtime backend: vLLM
  
- [View Full Benchmark Report](https://bench.trueperformancenetwork.com/orders/04250ff2-c3a8-4752-89e7-449944ebf542)

> **Note:**

> This is an initial evaluation result. A comparison

> against the original base model is needed to measure

> the actual impact of GRPO fine-tuning.


🚀 Deployment

1️⃣ Using Docker (Recommended)

The project can be run entirely using Docker—no need to manually install dependencies or run FastAPI/Streamlit separately.

Build the Docker image:

docker build -t grpo-smollm2 .

Run the Docker container:

docker run -p 8501:8501 --name grpo-smollm2-container grpo-smollm2


2️⃣ FastAPI (Backend)

The FastAPI service:

Loads the fine-tuned model

Exposes a REST endpoint for text generation

Designed for easy integration with any frontend

Run locally:

uvicorn api.main:app --host 0.0.0.0 --port 8000

3️⃣ Streamlit (Frontend)

The Streamlit app:

Provides a simple UI for prompt input

Displays model responses in real time

Communicates with the FastAPI backend

Run locally:

streamlit run streamlit_app/app.py

🧪 Use Cases

Length-controlled summarization

RLHF / RLAIF research experiments

Lightweight LLM deployment demos

Educational projects for GRPO & TRL

🧑‍💻 Contributing

Contributions and issues are welcome!

Please fork the repository and create pull requests for improvements.

Note: For portfolio and demo purposes, the FastAPI backend is run within the Streamlit application process.

Future Improvements:

Separate backend deployment

Containerized FastAPI service

Authentication & rate limiting

👤 Author

Beshoy Arnest

GitHub: https://github.com/7BeshoyArnest

LinkedIn: https://www.linkedin.com/in/beshoy-arnest-a3548a23a/
