
## Using OpenPrompt and OpenDelta on Big Models.

In this homework, we will show you how to use an OPT-2.7b (2.7 billion parameters) for direct question answering on a single 3090 GPU.

## Requirements
```
openprompt==1.0.1
opendelta==0.2.3
```

## TODOs

1. You may download the models and dataset using the bash scripts we provided.

2. You should complete the missing code in `optqa_to_complete.py` to train a delta model. By default it's the adapter, you can try other delta models on your own.

    If you done it correctly, finally you can train you model by shell command like
    ```
    python optqa_to_complete.py --plm_path ./plm_cache/opt-2.7b
    ```

    Hint: you can start with a smaller version of the OPT, e.g., opt-350m, which will make debugging much easier.

3. After you complete the training. You can play with your trained model with command line argument
`--mode interactive_inference`.


## Hand-in files:
1. The completed code.
2. The log of a successful training.
3. A brief report (in any format), which contains
```
1. Your delta checkpoint size comparaed to the backbone model (OPT) size.
2. The largest size of the OPT model you can use with delta tuning (If you complete the code in a correct way, it will be opt-2.7b) and the run time GPU memory.
3. Use a smaller OPT model, compare the GPU memory with and without delta tuning.
4. (Optional) The questions you've asked your model and its answers.
```

### FAQs
1. The disk space is used up when I download big models.
   The pre-installed anaconda is big. You can save space by:

    ```
    cd ~/anaconda3/
    rm -rf pkgs # delete the pkgs that anaconda download by default.
    ```
    and
    ```
    cd ~/anaconda3/envs
    du -h -d 1 # check which env takes up large space much and not used often.
    conda env remove -n ENVNAME # deleta
    ```
