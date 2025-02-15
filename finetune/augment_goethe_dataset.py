import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# === CONFIGURATION ===
MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit"
INPUT_FILE = "askagerman_goethe.json"
OUTPUT_FILE = "askagerman_goethe_augmented.json"
MAX_NEW_TOKENS = 250  # Increase for longer responses
BATCH_SIZE = 1  # Sequential processing for stability

# === LOAD MODEL & TOKENIZER ===
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    load_in_4bit=True,  # Use bitsandbytes 4-bit quantization
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # Avoid padding issues

# === PROMPT TEMPLATE ===
PROMPT_TEMPLATE_GOETHE = """You are an AI assistant specializing in German culture, helping students and newcomers understand German life. You must **first analyze** the provided formal answer from a Guide to German culture, customs and etiquette from Goethe University Frankfurt am Main for useful insights before formulating your response. 

**Guidelines:**
1. Keep the original answer's key points.
2. Phrase your answer in a clear, simple, and natural tone.
3. Keep your final response concise.
4. Look for and go through reasoning steps to support the formal answer.

### User Question:
{user_question}

### Original formal Answer:
{original_answer}

### Final Refined Answer:
<think>
"""

# === LOAD DATASET ===
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# === PROCESS DATASET ===
augmented_data = []

for item in tqdm(dataset, desc="Processing", total=len(dataset)):
    user_input = item["input"]
    original_answer = item["output"]

    # Format the prompt
    prompt = PROMPT_TEMPLATE_GOETHE.format(user_question=user_input, original_answer=original_answer)

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)

    # Generate output
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, 
            pad_token_id=tokenizer.eos_token_id, 
            temperature=0.7,  # Lower values = more deterministic, higher = more creative
            top_p=0.9  # Reduces extreme randomness
        )
        #output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id)
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Save augmented entry
    augmented_data.append({
        "input": user_input,
        "original_output": original_answer,
        "augmented_output": generated_text
    })

    # Optional: Save every 100 iterations to avoid data loss
    if len(augmented_data) % 10 == 0:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
            json.dump(augmented_data, out_f, indent=2, ensure_ascii=False)

# === SAVE FINAL OUTPUT ===
with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
    json.dump(augmented_data, out_f, indent=2, ensure_ascii=False)

print(f"Augmented dataset saved to {OUTPUT_FILE}")
