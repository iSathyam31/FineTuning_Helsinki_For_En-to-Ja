---
library_name: transformers
license: apache-2.0
base_model: Helsinki-NLP/opus-mt-en-jap
tags:
- translation
- generated_from_trainer
datasets:
- kde4
metrics:
- bleu
model-index:
- name: sattu-finetuned-kde4-en-to-jap
  results:
  - task:
      name: Sequence-to-sequence Language Modeling
      type: text2text-generation
    dataset:
      name: kde4
      type: kde4
      config: en-ja
      split: train
      args: en-ja
    metrics:
    - name: Bleu
      type: bleu
      value: 20.727494887708588
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# sattu-finetuned-kde4-en-to-jap

This model is a fine-tuned version of [Helsinki-NLP/opus-mt-en-jap](https://huggingface.co/Helsinki-NLP/opus-mt-en-jap) on the kde4 dataset.
It achieves the following results on the evaluation set:
- Loss: 1.4199
- Model Preparation Time: 0.0018
- Bleu: 20.7275

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 8
- eval_batch_size: 16
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 3
- mixed_precision_training: Native AMP

### Training results



### Framework versions

- Transformers 4.47.0
- Pytorch 2.5.1+cu124
- Datasets 3.1.0
- Tokenizers 0.21.0
