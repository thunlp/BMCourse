# L2_NLP_pipeline_pytorch
In this exercise, you are expected to construct a training pipeline based on pytorch and train on the glue-sst2, a sentiment analysis task. 
We have provided dataset in data/.
You are expected to complete .py files as follows.
## main.py
Some necessary module are as follows.   
1.batchify data   
2.build model   
3.train model   
4.evaluate model   
5.test model

## data.py
To complete a class that you can construct a dictionary from the dataset and get word2id and id2word. id refers to the position in the dictionary of a word.

## model.py
Some necessary module are as follows.   
1.__init__(). You can choose RNN or other model you are interested in.   
2.forward(). 

## Hand-in files (For registered students only)
1.The bash script(s).   
2.main.py  data.py  model.py  
3.a pdf that record train loss, valid loss, valid acc, test loss, test acc.
