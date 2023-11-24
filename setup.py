from setuptools import setup, find_packages

setup(name='gym_trade', 
      version='1.0',
    install_requires=[
        'gym<=0.23.1', 
        # 'opencv-python==4.2.0.34',
        'ruamel.yaml',
        'pandas',
        'numpy',
        'matplotlib',
        'mplfinance',
        'tqdm',
        'pandas_ta',
        'lightweight-charts',
        'pynput',
        'PyGObject',
        'pywebview[qt]',
        ], 
      packages=find_packages())
