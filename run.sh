# download code
# require PyTorch version >= 1.6.0
git clone --depth=1 -b backup  https://github.com/gzhch/fairseq.git code
cd code
pip install --editable ./
python setup.py build_ext --inplace
pip install scipy

# download roberta
wget https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz
tar -zxvf roberta.large.tar.gz

# download and preprocess glue
mkdir glue
cd glue
wget https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py
python download_glue_data.py --data_dir glue_data --tasks all
../examples/roberta/preprocess_GLUE_tasks.sh glue_data ALL
cd ..

# run code
# log和checkpoint默认保存在./tmp/out
chmod 777 run_roberta_local_stable.sh
CUDA_VISIBLE_DEVICES=1 ./run_roberta_local_stable.sh MRPC 1e-5 1

