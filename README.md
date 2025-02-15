# DeepseekGermanCultureGuide
A Telegram-Bot utilizing a distilled DeepSeekR1 model, adapted with additional information using LoRA

This project aims to create a chatbot, that is informed about german culture.
It is based on a distilled DeepSeek-R1 model, ```unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit```

We trained a LoRA adapter using unsloth.
## Training
### Data
The first part of our training data has been collected from scraped reddit data from the subreddit r/askagerman downloaded from https://the-eye.eu/redarcs/

The comments and posts were filtered to only include highly scored questions and answers by regular users.

The second part is based on generated Q&A pairs based on a German Customs guide by the Goethe University of Frankfurt.

### Augmentation
Both data sources include only questions and answers, so we generated the Chain of Thought reasoning to augment the data.
This process can be seen in ```finetune\augment_goethe_dataset.py``` and ```finetune\augment_reddit_dataset.py```.

The augmented json files are then combined and balanced to a dataset in the file ```finetune\prepare_dataset.py```

### LoRA
We train an adapter with rank 16 using unsloth see ```finetune\lora_training.py```
We then convert the finished safetensors-file to GGUF and upload to huggingface.

A variety of models is available at https://huggingface.co/johannesromisch, but we use ```FROM hf.co/johannesromisch/culture_model:latest``` as the final model.

## Usage in Ollama
We pull the model from ollama and add a System Prompt by building from out modelfile in ```models/modelfile```.

```python
ollama create culture14B_rank16_CoT_Data_simpleprompt -f models/modelfile
```

This should align the model with its primary objective of functioning as an assistant.

## Telegram as UI
For the final step, we use the ollama telegram API https://github.com/ruecat/ollama-telegram and tweak it slighty to exclude the reasoning part enclosed by think tags from the final answer.
Config is done in ```ollama-telegram\.env```