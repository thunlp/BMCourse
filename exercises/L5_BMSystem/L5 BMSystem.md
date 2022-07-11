# L5 Text Generation Demo based on BMInf

TA has already shown how to use BMInf in the lecture, which supports big model inference on a GTX 1060. This package supports three different big models, including CPM-1, CPM-2, and EVA. Hence, we can build various text generation applications on top of BMInf.

## 1. Install BMInf

Follow the instructions given by the BMInf repository.

## 2. Explore three big models

CPM-1 is an auto-regressive model and is good at generating text from left to right. CPM-2 is for blank filling. EVA is a dialog generation model.

Since BMInf will automatically download model checkpoints, you can load a model by a single line, such as `bminf.models.CPM2()`. Try some samples to understand the characteristics of each model. 

## 3. Design your application

In this project, you need to write templates for your application. Given a text input and your templates, the model needs to generate based on users' intent. Since you cannot update the model parameters, the template design is the only way to build your application.

[Here](https://github.com/OpenBMB/BMInf-demos) is some applications examples.

## Submission

You are required to submit a zip file containing the following files:

+ Your code (`.py` or `.ipynb`)
+ Good examples (`results.txt`), including the input texts and generated texts

Rename the zip file as `id_name.zip`, such as `2021010101_张三.zip`.
