# L3 Transformers

TA have already shown how to use 🤗 [Transformers](https://github.com/huggingface/transformers) in the lecture. However, you can't really learn how to use 🤗 Transformers just by watching the TA demonstrate the code. You need to actually get your hands dirty with 🤗 Transformers! 

Therefore in this exercise, you are expected to use the 🤗 Transformers to fine-tune a pre-trained language model yourself. You can refer to the [demo code](https://colab.research.google.com/drive/1tcDiyHIKgEJp4TzGbGp27HYbdFWGolU_#scrollTo=hB3IyMO6mWsA) given in our class if you run into trouble. 

You can simply edit the demo code to complete this exercise, **however**, we encourage you to first understand the demo code and then write your code without referring to it.

## 1. Learn to Explore the Model Hub
One of the greatest benefits of 🤗 Transformers is that it provides a wide variety of more than 50,000 models in its [model hub](https://huggingface.co/models). In this exercise, you are required to explore the model hub, and **load the [Tiny BERT](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2) in your code**. 

> Hint: copy the model's name ("google/bert_uncased_L-2_H-128_A-2"), and load the model with `from_pretrained()` just like that in the demo code.

Since TinyBERT is a really tiny, the performance won't be satisfaying (final accuracy only reaches 60~70%). You can browse the ModelHub and find other big models to fine-tune if you like. 

## 2. Load and Tokenize the Dataset
You are required to load and tokenize the QNLI dataset, which can be also easily done with `datasets` package. 

> 7.11 Update: The tokenizer for TinyBERT may not truncate the text appropriately. You need to add a keyword argument `max_length=512` when tokenizing the dataset.

## 3. Fine-tune the Model
Use the `Trainer` to fine-tune your Tiny BERT. The `TrainingArguments` can be kept the same with that in the demo code.

## 4. Report the Final Results
Report the best metric on the validation set. Note that normally, we need to select the model that works best on the validation set and report its results on the test set. However, since the test sets in GLUE benchmark do not include the ground-truth labels, to get the results on test set we need to submit the predictions to the official website. It is cumbersome, so we only need you to provide the best results on the validation set.

## Submission
You are required to submit a zip file containing the following files:

+ Your code (`.py` or `.ipynb`)
+ The best result (`result.txt`)

Rename the zip file as `id_name.zip`, such as `2021010101_张三.zip`.
