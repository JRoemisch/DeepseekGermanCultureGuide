# %% [markdown]
# Based on
# https://colab.research.google.com/drive/1AZghoNBQaMDgWJpi4RbffGM1h6raLUj9?usp=sharing#scrollTo=QYvyvuj5vd7H

# %%
import wandb
from unsloth import FastLanguageModel
from transformers import TrainingArguments

# %%
from unsloth import FastLanguageModel
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit",
    # model_name = "unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# %%
model = FastLanguageModel.get_peft_model(
    model,
    r = 8, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 8,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# %%
from datasets import Dataset
from datasets import concatenate_datasets
import json

culture_prompt_format = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Answer the following question trying to help a foreign tourist, exchange student, immigrant, expat or someone who is interested in German culture. Give short and clear answers.
{}

### Input:
{}

### Response:


{}"""

with open("datasets/askagerman_augmented_small_1121_handfiltered.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert JSON objects to prompts
reddit_formatted_data = [
    {"text": culture_prompt_format.format(item["input"], "", item["original_output"].split("<think>", 1)[1])} # Only use the output after the "<think>" tag
    # {"text": culture_prompt_format.format(item["input"], "", item["original_output"])}
    for item in data
]

# Convert to Hugging Face Dataset
reddit_dataset = Dataset.from_list(reddit_formatted_data)

#%%
with open("datasets/askagerman_goethe_augmented.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert JSON objects to Alpaca prompts
goethe_formatted_data = [
    {"text": culture_prompt_format.format(item["input"], "", item["augmented_output"].split("<think>", 1)[1])} # Only use the output after the "<think>" tag
    # {"text": culture_prompt_format.format(item["input"], "", item["original_output"])} 
    for item in data
]

# Convert to Hugging Face Dataset
goethe_dataset = Dataset.from_list(goethe_formatted_data)

#%%
# Add synthetic Data without reasoning to smaller goethe dataset to integrate into training.
# with open("datasets/ChatGPT_Dataset.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# # Convert JSON objects to Alpaca prompts
# chatgpt_formatted_data = [
#     {"text": culture_prompt_format.format(item["input"], "", item["output"])}
#     for item in data
# ]

# # Convert to Hugging Face Dataset
# chatgpt_dataset = Dataset.from_list(chatgpt_formatted_data)

# # concatenate datasets
# goethe_dataset = concatenate_datasets([goethe_dataset, chatgpt_dataset])


#%%
# Determine dataset sizes
len_reddit = len(reddit_dataset)
len_goethe = len(goethe_dataset)

# Oversample smaller datasets to match the largest one
num_repeats = len_reddit // len_goethe  # How many times we need to repeat Goethe
remainder = len_reddit % len_goethe  # If not evenly divisible, take extra samples

goethe_repeated = concatenate_datasets([goethe_dataset] * num_repeats)  # Repeat fully
if remainder > 0:
    goethe_repeated = concatenate_datasets([goethe_repeated, goethe_dataset.select(range(remainder))])  # Add extra samples

# Combine datasets
balanced_dataset = concatenate_datasets([reddit_dataset, goethe_repeated]).shuffle(seed=42)

# %%
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = balanced_dataset, 
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = True, # Packs short sequences together to save time!
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_ratio = 0.1,
        num_train_epochs = 1,
        learning_rate = 2e-5,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.1,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "wandb", # Use this for WandB etc
        run_name = "lora_training_german_culture_bot"
    ),
)

wandb.init(project="lora_training_german_culture_bot", name="7B Balanced Reddit&Goethe")

# %%
trainer_stats = trainer.train()

# %% [markdown]
# ## Saving, loading finetuned models
# 
# To save the final model as LoRA adapters, either use Huggingface's push_to_hub for an online save or save_pretrained for a local save.
# 
# [NOTE] This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!

# %%
model.save_pretrained("lora_model_goethe_reddit_balanced_cot") # Local saving
tokenizer.save_pretrained("lora_model_goethe_reddit_balanced_cot")


# Convert to GGUF and push to Huggingface (This was done in Google Colab, due to environment and disk space limitations)
# model.push_to_hub_gguf("johannesromisch/culture_model7B", tokenizer, quantization_method = "q4_k_m", token = "my_token")