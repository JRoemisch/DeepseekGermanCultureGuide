# DeepseekGermanCultureGuide
A Telegram-Bot utilizing a distilled DeepSeekR1 model, adapted with additional information using LoRA

This project aims to create a chatbot, that is informed about german culture.
It is based on a distilled DeepSeek-R1 model, ```unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit```

We trained a LoRA adapter using unsloth.
## Training
### Data
The first part of our training data has been collected from scraped reddit data from the subreddit r/askagerman downloaded from https://the-eye.eu/redarcs/

The comments and posts were combined and filtered to only include highly scored questions and answers (score greater than 50) by regular users.
See ``datasets\build_data_from_scraped_reddit.py``

The second part is based on generated Q&A pairs based on a German Customs guide by the Goethe University of Frankfurt.

### Augmentation
Both data sources include only questions and answers, so we generated the Chain of Thought reasoning to augment the data.
This process can be seen in ```finetune\augment_goethe_dataset.py``` and ```finetune\augment_reddit_dataset.py```, which results in the files in ``finetune\datasets``.

### LoRA
For the final approach, the augmented json files ``askagerman_augmented_small_1121_handfiltered.json`` and ``askagerman_goethe_augmented.json`` are combined and balanced by repeating the goethe-data to match the number of training samples.

We fine-tune an adapter with rank 16 using unsloth see ```finetune\lora_training.py```
We then convert the safetensors-file to GGUF and upload to huggingface.

The final model can be found and tested at https://huggingface.co/johannesromisch/culture_model.

## Usage in Ollama
We pull the model from ollama and add a System Prompt by building from our modelfile ```modelfile```.

```python
ollama create culture14B_rank16_CoT_Data_simpleprompt -f models/modelfile
```

This should align the model with its primary objective of functioning as an assistant.

## Telegram as UI
For the final step, we use the ollama telegram API https://github.com/ruecat/ollama-telegram and tweak it slighty to exclude the reasoning part enclosed by think tags from the final answer.
Config is done in ```ollama-telegram\.env```

# Models
All models are distilled DeepseekR1 adaptations.

#### ``https://huggingface.co/johannesromisch/culture_model:``
Final model 14B

- base_model: unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit
- LoRA Rank: 16
- Quantized: 4B, (15GB Q8 version available as GGUF-file download; was not used.)
- Reddit and Goethe-Uni data augmented with CoT  

#### ``https://huggingface.co/johannesromisch/culture_model7B:``
Smaller 7B version of Final model $\to$ faster

- base_model: unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit

#### ``https://huggingface.co/johannesromisch/7B_ALLData_Culture:``  
Lower rank adaptation, smaller 7B model, added synthetic data, without CoT augmentation
- base_model: 
unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit
- LoRA Rank: 8
- Quantized: 4B, 5B, 8B
- Reddit, Goethe-Uni and synthetic ChatGPT Q&A data, not augmented
