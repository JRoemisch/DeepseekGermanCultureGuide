import json
from datasets import Dataset
from dotenv import load_dotenv
import os

load_dotenv()

# Access environment variables
alpaca_prompt = os.getenv("ALPACA_PROMPT")
reddit_prompt = os.getenv("REDDIT_PROMPT")

model_name = os.getenv("MODEL_NAME")
dataset_name = os.getenv("DATASET_NAME")
gen_dataset_name = os.getenv("GEN_DATASET_JSON_NAME")
reddit_dataset_json_name = os.getenv("REDDIT_DATASET_JSON_NAME")
reddit_dataset_name = os.getenv("REDDIT_DATASET_NAME")

# Load JSON file
# with open("ChatGPT_Dataset.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

with open(reddit_dataset_json_name, "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert JSON objects to Alpaca prompts
formatted_data = [
    {"text": reddit_prompt.format("", item["input"], item["output"])}
    for item in data
]

# Convert to Hugging Face Dataset
dataset = Dataset.from_list(formatted_data)

# Save the dataset (optional)
dataset.save_to_disk(reddit_dataset_name)