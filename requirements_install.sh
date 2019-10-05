cd /tmp
curl -O https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
source ~/.bashrc
cd ~
conda create -n tensorflow_env tensorflow
conda activate tensorflow_env
conda install -c conda-forge librosa
