ENVNAME=new_env
conda create -n $ENVNAME python=3.8 &&
source ~/anaconda3/etc/profile.d/conda.sh &&
conda activate $ENVNAME &&


rm -rf torch-1.11.0+cu115-cp38-cp38-linux_x86_64.whl
wget https://download.pytorch.org/whl/cu115/torch-1.11.0%2Bcu115-cp38-cp38-linux_x86_64.whl &&
pip install torch-1.11.0+cu115-cp38-cp38-linux_x86_64.whl &&


# before git clone, add your ssh key to github.com
rm -rf word2vec-pytorch
git clone git@github.com:ShengdingHu/word2vec-pytorch.git &&
cd word2vec-pytorch &&
pip install -r requirements.txt &&


rm -rf weights/ &&
python train.py --config config.yaml  # the first time might be slow
