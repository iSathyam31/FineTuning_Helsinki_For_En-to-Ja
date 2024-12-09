## Fine-Tuned English-to-Japanese Translation Model

This repository contains a fine-tuned version of the [Helsinki-NLP/opus-mt-en-jap](https://huggingface.co/Helsinki-NLP/opus-mt-en-jap) model for English-to-Japanese translation, trained using the KDE4 dataset.

The fine-tuning process enhances translation quality, specifically for domain-specific sentences found in KDE software documentation and similar contexts.

## Features
- Enhanced English-to-Japanese translation for technical and software-related contexts.
- Fine-tuned using the [KDE4 dataset](https://huggingface.co/datasets/kde4) for improved domain relevance.

## Model
The fine-tuned model is hosted on Hugging Face: [sattu-finetuned-kde4-en-to-jap](https://huggingface.co/iSathyam03/sattu-finetuned-kde4-en-to-jap).

## How to Use
You can load the model directly from Hugging Face:

```
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("iSathyam03/sattu-finetuned-kde4-en-to-jap")
model = AutoModelForSeq2SeqLM.from_pretrained("iSathyam03/sattu-finetuned-kde4-en-to-jap")

# Translate text
text = "This is a sample English sentence."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
outputs = model.generate(**inputs)
translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(translated_text)
```

## Training Process
#### Dataset
 * **KDE4 Dataset**: Contains parallel English-Japanese sentences from KDE software documentation. This dataset ensures the translations are relevant to technical and domain-specific contexts.

#### Training Arguments
The model was fine-tuned using the following `Seq2SeqTrainingArguments`:
```
from transformers import Seq2SeqTrainingArguments

args = Seq2SeqTrainingArguments(
    f"sattu-finetuned-kde4-en-to-jap",
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=True,
)
```

#### Framework
**Base Model**: Helsinki-NLP/opus-mt-en-jap.
**Libraries**: Transformers and PyTorch.


## Evaluation Metrics

The fine-tuning process significantly improved the model's performance, as shown by the evaluation metrics:

| **Metric**                  | **Before Fine-Tuning** | **After Fine-Tuning** |
|-----------------------------|------------------------|------------------------|
| **Evaluation Loss**         | 10.38                 | 1.42                  |
| **BLEU Score**              | 0.0046                | 20.73                 |
| **Evaluation Runtime (s)**  | 522.76                | 503.29                |
| **Samples per Second**      | 25.14                 | 26.11                 |
| **Steps per Second**        | 1.57                  | 1.63                  |

- **BLEU Score**: Improved from near-zero (0.0046) to 20.73, reflecting the enhanced translation quality for English-to-Japanese.
- **Evaluation Loss**: Reduced from 10.38 to 1.42, showing the model's increased accuracy.
- Other metrics, such as runtime and throughput, show slight improvements due to fine-tuning optimizations.


## Constraints and Recommendations
This project was developed using limited hardware resources, specifically an RTX 4050 GPU with 6GB of VRAM. While this setup worked for fine-tuning, users may face constraints when working with larger models or datasets.

If you're using limited resources, consider the following recommendations to optimize your training process. Alternatively, you can use platforms like Google Colab or more powerful hardware setups for a smoother experience.

#### Tips for Working with Limited Resources
1. Reduce `per_device_train_batch_size`: A large batch size can exceed your GPU's memory capacity. Reduce it as needed.
```
per_device_train_batch_size=8  # Reduce to 16 or 8 if needed
```
2. Reduce `per_device_eval_batch_size`: Similarly, reduce the evaluation batch size to save memory.
```
per_device_eval_batch_size=16  # Reduce to 16 or 8
```
3. Use `gradient_accumulation_steps`: Simulate a larger batch size by accumulating gradients across steps
```
gradient_accumulation_steps=2  # Simulate larger batch size
```
4. Enable Mixed Precision Training (`fp16`): Mixed precision training reduces memory usage by using 16-bit floating-point precision
```
fp16=True  # Already enabled in this project
```
5. Remove Unused Columns and Efficient Padding: If your dataset contains unused columns, remove them to save memory. Set `label_pad_token_id` for text generation tasks.
```
remove_unused_columns=True
label_pad_token_id=-100  # For text generation or seq-to-seq tasks
```
6. Use `save_steps` instead of `save_strategy="epoch"`: If memory usage is a concern, saving the model after every epoch can lead to frequent storage usage. You can save more frequently using save_steps.
```
save_steps=500  # Save every 500 steps or as needed
```
7. Limit `push_to_hub`: Pushing to the hub might be memory-intensive. If you're not actively pushing to the hub during training, consider disabling it until you're ready.
```
push_to_hub=False  # Turn off unless you need to push to Hugging Face Hub during training
```
#### Example Updated Code:
```
from transformers import Seq2SeqTrainingArguments

args = Seq2SeqTrainingArguments(
    output_dir="sattu-finetuned-kde4-en-to-jap",
    evaluation_strategy="no",
    save_strategy="epoch",  # Or save_steps=500 if you prefer more frequent saving
    learning_rate=2e-5,
    per_device_train_batch_size=16,  # Reduced batch size
    per_device_eval_batch_size=32,  # Reduced evaluation batch size
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,  # Mixed precision training enabled
    push_to_hub=False,  # Turned off unless needed
    gradient_accumulation_steps=2,  # Accumulate gradients for larger effective batch size
    remove_unused_columns=True,  # Remove unused columns to save memory
    label_pad_token_id=-100  # Used in text generation tasks
)
```


The Jupyter Notebook detailing the training process is included: [code.ipynb](code.ipynb)

## Dependencies
Install the required dependencies using:
```
pip install -r requirements.txt

```

## Acknowledgements
* Hugging Face and the Transformers library.
* KDE4 dataset for domain-specific parallel data.
* Helsinki-NLP for the base translation model.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
