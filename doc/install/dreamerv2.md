## 1. Install dreamerv2 environment


### 1.1. Initialize your environment
```sh
source ./init.sh
```


### 1.2. Install
```sh
conda install cudnn=8.2 cudatoolkit=11.3 -c anaconda -y # install cuda-toolkit for gpu support
conda install ffmpeg -y
pip install tensorflow==2.9.0 tensorflow_probability==0.17.0 protobuf==3.20.1
pushd ext/dreamerv2/
pip install -e .
popd
```

### 1.3. Test Installation

Check gpu in python intepreter

```sh
source ./init.sh
python -c 'import tensorflow as tf; tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)'
```





## 2. Install in Colab (Optional)

```
from google.colab import drive
drive.mount('/content/drive')
```
```
!wget -c https://repo.anaconda.com/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh
!ls
!chmod +x Miniconda3-py37_4.10.3-Linux-x86_64.sh
%env PYTHONPATH=
!./Miniconda3-py37_4.10.3-Linux-x86_64.sh -b -f -p /usr/local
import sys
_ = (sys.path.append("/usr/local/lib/python3.7/site-packages"))
!conda install cudnn=8.2 cudatoolkit=11.3 -c anaconda -y # install cuda-toolkit for gpu support
!conda install ffmpeg -y
!pip install tensorflow==2.9.0 tensorflow_probability==0.17.0 protobuf==3.20.1
!git clone https://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@github.com/linhongbin-ws/gym-ras.git
%cd ./gym-ras
!git submodule update --init --recursive
%cd ext/dreamerv2/
!pip install -e .
%cd ../..
!pip install pybullet==3.0.9
%cd ext/SurRoL/
!pip install -e .
%cd ../..
!pip install -e .
!pip install pybullet==3.0.9
!conda install -c conda-forge libstdcxx-ng -y
```
```
!python gym_ras/run/train.py --baseline dreamerv2 --baseline-tag gym_ras_np --logdir /content/drive/MyDrive/colab-tf-log
``` -->

<!-- ```
import tensorflow as tf
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
sys_details = tf.sysconfig.get_build_info()
cuda_version = sys_details["cuda_version"]
print(cuda_version)
```