from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel
import torch

app = FastAPI()

class InputText(BaseModel):
    prompt: str
    max_new_tokens: int = 96  

# Load tokenizer and model as specified in the notebook 
model_id = "7beshoyarnest/fine_tuned_SmolGRPO-135M_using_GRPO" 
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Initialize model with notebook-specific optimizations 
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
except Exception:
    # Fallback if flash_attention_2 is not supported on the host machine 
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto"
    )

model.eval()

@app.post("/generate")
def generate_response(input: InputText):
    if not input.prompt or input.prompt.strip() == "":
        raise HTTPException(
            status_code=400, 
            detail="No prompt provided."
        )
    
    # Format input for an instruct model if necessar
    inputs = tokenizer(input.prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=input.max_new_tokens,
            do_sample=True,
            temperature=0.7
        )

    # Decode only the generated part (excluding the original prompt)
    generated_text = tokenizer.decode(
        output_tokens[0][inputs['input_ids'].shape[-1]:], 
        skip_special_tokens=True
    )

    return {
        "prompt": input.prompt, 
        "generated_text": generated_text
    }