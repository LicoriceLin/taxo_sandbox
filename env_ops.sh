sudo apt-get update 
sudo apt-get install g++

conda activate n4jn
conda install ipython
pip3 install torch torchvision torchaudio
pip install lightning datasets torchmetrics transformers "lightning[pytorch-extra]" captum tensorboard
pip install numpy pandas scipy matplotlib seaborn biopython networkx

# not essential
pip install neomodel fastcluster pygraphviz tbparse
conda install conda-forge::pymol-open-source