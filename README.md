<!-- PASTE INTO: github.com/Suyash018/English-to-Hinglish-fintuned-lamma-3-8b-instruct → README.md -->
<!-- If you rename the repo (recommended: Llama-3-8B-English-to-Hinglish-LoRA), GitHub redirects the old URL automatically. -->

# English → Hinglish Translation with Llama 3 8B (LoRA)

A Llama 3 8B Instruct model fine-tuned with QLoRA to translate standard English into **Hinglish** — the Hindi-English code-mixed register used in everyday informal communication across India. Model weights and dataset are hosted on Hugging Face.

**🤗 Model:** [suyash2739/English_to_Hinglish_fintuned_lamma_3_8b_instruct](https://huggingface.co/suyash2739/English_to_Hinglish_fintuned_lamma_3_8b_instruct) — 3,000+ cumulative downloads across variants
**🤗 Dataset:** [suyash2739/News_Hinglish_English](https://huggingface.co/datasets/suyash2739/News_Hinglish_English) — published with DOI [10.57967/hf/5120](https://doi.org/10.57967/hf/5120)

## Why this exists

Hinglish is one of the most widely used code-switched registers in the world, yet most translation systems target formal Hindi (Devanagari), not the romanized, code-mixed way people actually write in chats, comments, and headlines. This project fine-tunes an open 8B model to produce natural Hinglish and releases both the model and the training dataset openly.

## Training

| | |
|---|---|
| Base model | `unsloth/llama-3-8b-Instruct-bnb-4bit` |
| Method | QLoRA (4-bit), Unsloth + HuggingFace TRL |
| Data | News_Hinglish_English (curated English ↔ Hinglish pairs, news domain) |
| Task format | Instruction-style: "Translate the input from English to Hinglish" |

Training loss curve:

![Loss curve](https://cdn-uploads.huggingface.co/production/uploads/65187b234965add2b08b2990/31vSqxldRSGEDNGwrJbFy.png)

## Qualitative comparison vs GPT-4o

**English input:**
> Finance Minister Nirmala Sitharaman said, "There used to be a poverty index...a human development index and all of them continue, but today what is keenly watched is VIX, the volatility index of the markets." Stability of the government is important for markets to be efficient, she stated. PM Narendra Modi's third term will make markets function with stability, she added.

**GPT-4o:**
> Finance Minister Nirmala Sitharaman ne kaha, "Pehle ek poverty index hota tha...ek human development index hota tha aur yeh sab ab bhi hain, lekin aaj jo sabse zyada dekha ja raha hai, woh hai VIX, jo markets ka volatility index hai." Unhone kaha ki sarkar ki stability markets ke efficient hone ke liye zaroori hai. PM Narendra Modi ka teesra term markets ko stability ke saath function karne mein madad karega, unhone joda.

**This model (fine-tuned Llama 3 8B):**
> Finance Minister Nirmala Sitharaman ne kaha, "Pehle ek poverty index hota tha... ek human development index hota tha aur sab kuch ab bhi chal raha hai, lekin aaj jo kaafi zyada dekha ja raha hai, woh VIX hai, jo markets ki volatility ka index hai." Unhone kaha ki markets ke liye sarkar ki stability zaroori hai. PM Narendra Modi ke teesre term se markets stability ke saath function karenge, unhone joda.

An 8B open model matching GPT-4o's fluency on this register at a fraction of the inference cost. Quantitative benchmarks (BLEU / chrF on a held-out test split) are in progress and will be published in the Hugging Face model card.

## Usage

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers trl peft accelerate bitsandbytes
```

```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="suyash2739/English_to_Hinglish_fintuned_lamma_3_8b_instruct",
    max_seq_length=2048,
    dtype=None,          # auto-detect; float16 for T4/V100, bfloat16 for Ampere+
    load_in_4bit=True,   # 4-bit quantization to reduce memory
)

def translate(text):
    prompt = """Translate the input from English to Hinglish to give the response.

### Input:
{}

### Response:
"""
    inputs = tokenizer([prompt.format(text)], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=2048, use_cache=True)
    raw = tokenizer.batch_decode(outputs)[0]
    return raw.split("### Response:\n")[1].split("<|eot_id|>")[0]

print(translate("This is a fine-tuned Hinglish translation model using Llama 3."))
# Yeh ek fine-tuned Hinglish translation model hai jo Llama 3 ka istemal karta hai.
```

Model weights live on Hugging Face (GitHub repo size limits); the notebook in this repo walks through fine-tuning and inference end to end.

## Limitations

- Trained primarily on news-domain text; casual/slang-heavy registers may be less natural.
- Outputs romanized Hinglish only (no Devanagari).
- Inherits biases of the base Llama 3 model and the source news corpus.

## Citation

If you use the dataset or model, please cite:

```bibtex
@misc{agarwal2024newshinglish,
  author    = {Agarwal, Suyash},
  title     = {News\_Hinglish\_English: An English--Hinglish Parallel Corpus},
  year      = {2024},
  publisher = {Hugging Face},
  doi       = {10.57967/hf/5120},
  url       = {https://huggingface.co/datasets/suyash2739/News_Hinglish_English}
}
```

## License

Apache 2.0
