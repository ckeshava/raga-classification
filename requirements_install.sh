cd /tmp
curl -O https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
bash Anaconda3-2019.03-Linux-x86_64.sh
source ~/.bashrc
printf "installed anaconda" 
cd ~
conda create -n tensorflow_env tensorflow
conda activate tensorflow_env
conda install -c conda-forge librosa
printf "installed librosa"

conda install -c conda-forge matplotlib
printf "installed matplotlib"

conda install -c menpo imageio
printf "installed image_io"
