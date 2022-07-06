



SCALE="2.7b" # select from 125m, 350m, 1.3b, 2.7b;  6.7b and 30b is too big for single 3090. See in https://huggingface.co/models?sort=downloads&search=opt
MODEL_PATH="./plm_cache"



mkdir $MODEL_PATH
cd $MODEL_PATH

git clone https://huggingface.co/facebook/opt-${SCALE}
cd opt-${SCALE}

if [ "${SCALE}" == "6.7b" ]; then
    echo "Downloading 6.7b model"
    rm -rf pytorch_model-0000*-of-00002.bin
    wget https://huggingface.co/facebook/opt-${SCALE}/resolve/main/pytorch_model-00001-of-00002.bin
    wget https://huggingface.co/facebook/opt-${SCALE}/resolve/main/pytorch_model-00001-of-00002.bin
else
    echo "Downloading model for ${SCALE}"
    rm -rf pytorch_model.bin
    wget https://huggingface.co/facebook/opt-${SCALE}/resolve/main/pytorch_model.bin
fi

