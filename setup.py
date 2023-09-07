from setuptools import setup, find_packages

setup(name='gym_ras', 
      version='1.0',
    install_requires=[
        'gym<=0.24', 
        'opencv-python==4.2.0.34',
        'ruamel.yaml',
        'pandas',
        'numpy',
        'matplotlib',
        'mplfinance',
        'tqdm',
        'pandas_ta',
        ], 
      packages=find_packages())
