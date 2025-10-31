pip install "git+https://github.com/slds-lmu/yahpo_gym#egg=yahpo_gym&subdirectory=yahpo_gym"
git clone https://github.com/slds-lmu/yahpo_data.git
mv yahpo_data/ hpo_benchmarks

conda install -y -c conda-forge optuna=3.6.0
conda install -y numpy=1.23.5 -c conda-forge
conda install -y pyDOE2=1.3.0 -c conda-forge
conda install -y omegaconf=2.3.0 -c conda-forge
conda install -y ipykernel=6.29.4 
conda install -y cmaes=0.10.0 -c conda-forge

conda install -y matplotlib
conda install -y pandas


pip install hpbandster
pip install black
pip install flake8
pip install moviepy
pip install opencv-python==4.6.0.66

pip install git+https://github.com/ZebinYang/SeqUD.git