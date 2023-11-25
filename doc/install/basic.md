# 1. Install

## 1.1. Create init bash file

```sh
touch ./init.sh
echo "ANACONDA_PATH="$HOME/anaconda3""  >> ./init.sh # modify according to your installation 
echo "ENV_NAME=gym_trade" >> ./init.sh
```

## 1.2. Install gym_trade

```sh
source ./init.sh
sudo apt install libgirepository1.0-dev -y
conda create -n $ENV_NAME python=3.9 -y
conda activate $ENV_NAME
conda install -c conda-forge libffi -y
pip install -e .
```

## 1.3. Add to init file

```sh
echo "source $ANACONDA_PATH/bin/activate" >> ./init.sh
echo "conda activate $ENV_NAME" >> ./init.sh
```


## 1.4. test installation

```sh
source ./init.sh
python ./test/env_test.py
```