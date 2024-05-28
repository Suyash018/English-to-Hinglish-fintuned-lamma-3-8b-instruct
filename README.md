


# Link to HuggingFace model

Hugging Face - [https://huggingface.co/suyash2739/English_to_Hinglish_fintuned_lamma_3_8b_instruct ]

# Project - A English to Hinglish Language Translater.
This Project aims to develop a high-performance language translation model capable of translating standard English to Hinglish (a blend of Hindi and English commonly used in informal communication in India).

This model is fine-tuned by me and Huggingface repo is linked above. Cannot upload model to github as repo size in github is limited.

ipynb file on how to use the repo is added

# Loss Curve


![image/png](https://cdn-uploads.huggingface.co/production/uploads/65187b234965add2b08b2990/31vSqxldRSGEDNGwrJbFy.png)

# Comaprision / Performance

- English
```python
English = """Finance Minister Nirmala Sitharaman said, "There used to be a poverty index...a human development index and all of them continue, but today what is keenly watched is VIX, the volatility index of the markets." Stability of the government is important for markets to be efficient, she stated. PM Narendra Modi's third term will make markets function with stability, she added."""
```
- Gpt 4o
```python
Gpt 4o = """ Finance Minister Nirmala Sitharaman ne kaha, "Pehle ek poverty index hota tha...ek human development index hota tha aur yeh sab ab bhi hain, lekin aaj jo sabse zyada dekha ja raha hai, woh hai VIX, jo markets ka volatility index hai." Unhone kaha ki sarkar ki stability markets ke efficient hone ke liye zaroori hai. PM Narendra Modi ka teesra term markets ko stability ke saath function karne mein madad karega, unhone joda."""
```

- My model (Finetuned LLama model)
```python
LLama model = Finance Minister Nirmala Sitharaman ne kaha, "Pehle ek poverty index hota tha... ek human development index hota tha aur sab kuch ab bhi chal raha hai, lekin aaj jo kaafi zyada dekha ja raha hai, woh VIX hai, jo markets ki volatility ka index hai." Unhone kaha ki markets ke liye sarkar ki stability zaroori hai. PM Narendra Modi ke teesre term se markets stability ke saath function karenge, unhone joda.
```


![image/png](https://cdn-uploads.huggingface.co/production/uploads/65187b234965add2b08b2990/Rc3nlfnSVwu1dnzfxYb-Y.png)



# Inference / How to use the model:

```
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps xformers trl peft accelerate bitsandbytes
```

```python
from unsloth import FastLanguageModel
import torch
max_seq_length = 2048
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "suyash2739/English_to_Hinglish_fintuned_lamma_3_8b_instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
```


```python

def pipe(text):
  prompt = """Translate the input from English to Hinglish to give the response.

### Input:
{}

### Response:
"""
  inputs = tokenizer(
      [
          prompt.format(text),
      ], return_tensors = "pt").to("cuda")

  outputs = model.generate(**inputs, max_new_tokens = 2048, use_cache = True)
  raw_text = tokenizer.batch_decode(outputs)[0]
  return raw_text.split("### Response:\n")[1].split("<|eot_id|>")[0]
```

```python
text = "This is a fine-tuned Hinglish translation model using Llama 3." # INPUT
print(pipe(text))
## Yeh ek fine-tuned Hinglish translation model hai jo Llama 3 ka istemal karta hai.
```



# Uploaded  model

- **Developed by:** suyash2739
- **License:** apache-2.0
- **Finetuned from model :** unsloth/llama-3-8b-Instruct-bnb-4bit

This llama model was trained 2x faster with [Unsloth](https://github.com/unslothai/unsloth) and Huggingface's TRL library.

[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20made%20with%20love.png" width="200"/>](https://github.com/unslothai/unsloth)#
