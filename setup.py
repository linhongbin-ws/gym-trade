from setuptools import setup, find_packages

setup(name='gym_trade', 
      version='1.0',
    install_requires=[
        'gym<=0.23.1', 
        # 'opencv-python==4.2.0.34',
        'ruamel.yaml<=0.17',
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
        'opencv-python-headless', # dont use opencv-python, https://forum.qt.io/topic/119109/using-pyqt5-with-opencv-python-cv2-causes-error-could-not-load-qt-platform-plugin-xcb-even-though-it-was-found/8
        ], 
      packages=find_packages())
