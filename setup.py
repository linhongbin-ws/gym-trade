from setuptools import setup, find_packages

setup(name='gym_trade', 
      version='1.0',
    install_requires=[
        'gym<=0.23.1', 
        'opencv-python', # <=4.1 , avoid conflit with pyqt5, https://blog.csdn.net/Torch_HXM/article/details/123807278
        'ruamel.yaml<=0.17',
        'pandas',
        'numpy',
        'matplotlib',
        'mplfinance',
        'yfinance',
        'tqdm',
        'pandas_ta',
        'lightweight-charts',
        'pynput',
        'pyqt',
        'PyGObject==3.50.0',
        'pywebview[qt]',
        'pandas-market-calendars==4.5',
        "pytz",
        ], 
      packages=find_packages())
