## Installation

<!-- 
```sh
conda create -n gym-trade python=3.9 -y
conda activate gym-trade
conda install anaconda::pyqtwebengine -y
pip install -e .
``` -->

uv install
```sh
uv venv --python 3.9
source .venv/bin/activate
uv pip install -e . -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

and add global env variable to .bashrc
```sh
export PYWEBVIEW_GUI=qt
```

## TODO

- tradingview render (https://github.com/louisnw01/lightweight-charts-python)

