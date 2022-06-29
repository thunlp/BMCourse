# L2_NLP_pipeline_pytorch
In this exercise, you are expected to construct a training pipeline based on pytorch and train on the glue-sst2, a sentiment analysis task.
We have provided the dataset in data/.
You are expected to complete three python files as follows.

## main.py
Some necessary modules are as follows.
1. load and batchify data
    Turn the whole dataset into batches. In each optimization step, we pass a batch to the model and get the loss, etc.)
2. build model
   1. Call the model initialization method.
3. train model
   1. Call the model's forward method in each step
4. evaluate model
   1. Call the evaluation function after each epoch
5. test model
   1. Test the model's performance and report the accuracy

## data.py
Write a class, you can construct a dictionary from the dataset and get word2id and id2word. id refers to the position in the dictionary of a word.

You can also use this scripts to send batch data to main.py

## model.py
Some necessary module are as follows.
1. __init__(). You can choose RNN or other model you are interested in.
2. forward(). The computation steps of your model.

## Hand-in files (For registered students only)
1. The bash script(s).
2. main.py  data.py  model.py
3. a pdf file that records train loss, valid loss, valid acc, test loss, test acc.


## Hint
1. We provide a [reference_code_and_results](reference_code_and_results.pdf) which contains the code snippet of the language modeling using RNN in the lecture. To encourage you coding line by line, we provide it in picture format.

2. The difference between this exercise and the task taught in the lecture is that:
   1. In this lecture it is a language modeling task, that is we predict the next word of the sentence. Which is glue-sst2, it is a binary classification tasks.
   2. You can use the last hidden state of your RNN model (if you use RNN) to do a binary classification using the CrossEntroy loss taught in the lecture.
   3. The evaluation doesn't use PPL (perplexity score), instead we use accuracy.

3. Finding example scripts from the web is also encouraged.