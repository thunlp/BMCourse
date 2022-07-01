# Getting Started with Simple Word2Vec Training

In this exercise, you are expected to adopt a public and straightforward realization of word2vec.
You will git clone that project, build your virtual environment, run the code through
by submitting bash scripts to a GPU server, and finally write simple bash scripts to do a small grid search and automatically summarize the result to one file.
All these kinds of stuff pave the way for your future research in the deep learning area.

## Git clone a public project.
The public project we will clone is [OlgaChernytska/word2vec-pytorch](https://github.com/OlgaChernytska/word2vec-pytorch). To keep the code, I also made a [copy](https://github.com/ShengdingHu/word2vec-pytorch) and made some modifications to the requirements. You can refer to it if you encounter difficulties in building the virtual environment. 
Please read the code to get an impression of the related knowledge learned in the lecture.


## Build a virtual environment
You are encouraged to build a separate virtual environment for each project to make
the code base clean. Please refer to the lecture on virtual environment

## Write simple bash scripts to run the code.
Bash is often used to automate the running of a pipeline. You should learn its basic commands.

Some suggestions are here: [for-loop](https://www.cyberciti.biz/faq/bash-for-loop/), [parallel computation](https://unix.stackexchange.com/questions/103920/parallelize-a-bash-for-loop), [passing arguments](https://www.baeldung.com/linux/use-command-line-arguments-in-bash-script),
[redirectingoutputs](https://www.redhat.com/sysadmin/redirect-operators-bash#:~:text=The%20append%20%3E%3E%20operator%20adds%20the,uname%20%2Dr%20to%20the%20specifications.).

For students who use cloud computation resources, please refer to the lecture for submitting jobs to the cluster.

**Hint:**
1. When you encounter `no module named xxx` error. Don't worry, installing the missing packages will solve the problem.
2. The `requirements.txt` is too strict in terms of package version. Later (or even earlier versions) may also work.

## Hyper-parameter search
Now pass different hyperparameters to the word2vec training to see how loss change. Please only tune the **learning rate** and **train_batch_size** for now. Since this homework is only a start-up project, do not search for over 10 groups of
hyper-parameter for saving computational resources.

You should create a summarized log file in any format (csv, json, excel, txt, etc.) as long as it can neatly keep track of your experiments. You should automatically log the **validation loss of the last epoch** and the corresponding **hyper-parameters values** to this single file for automatic comparison. You can accomplish it either in bash or by modifying the python code.

**Hint:**
1. To change the learning rate, etc., you may use the `argparse` package. Or you can modify the config.yaml using some function. It's up to you.

## Hand-in files (For registered students only)

1. The bash script(s) for hyperparameter search,
2. The summarized log
3. Any python/yaml file you modified compared to the original word2vec-pytorch projects.

Hand in to web-learning platform.


## Acknowledgement
We thank [OlgaChernytska](https://github.com/OlgaChernytska) for providing a simple realization of the Word2Vec as the starter code for deep learning enthusiasts.
