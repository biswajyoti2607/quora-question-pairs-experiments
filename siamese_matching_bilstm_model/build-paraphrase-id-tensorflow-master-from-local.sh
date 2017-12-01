sudo apt-get -y install software-properties-common
sudo apt-add-repository -y universe
sudo apt-get -y update
sudo apt-get -y install python-pip
sudo apt-get -y install python3-pip
sudo apt-get -y install unzip
unzip paraphrase-id-tensorflow-master.zip
cd paraphrase-id-tensorflow-master/
pip install -r requirements.txt
pip3 install -r requirements.txt
python -m nltk.downloader punkt
python3 -m nltk.downloader punkt
make aux_dirs
make glove
cd scripts/run_model/../../data/external/
unzip glove.6B.zip
cd ../../
read -n1 -r -p "Upload data files and Press space to continue..." key
make quora_data
nohup python3 scripts/run_model/run_siamese_matching_bilstm.py train --share_encoder_weights --model_name=siamese_matching --run_id=0 &