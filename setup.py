from setuptools import setup, find_packages

install_requires = [
    'gym<=0.23.1',
    'opencv-python',
    'ruamel.yaml<=0.17',
    'pandas',
    'numpy',
    'matplotlib',
    'mplfinance',
    'yfinance',
    'tqdm',
    'hydra-core',

    # ⬇️ SSH Git dependency（正确写法）
    # 'pandas-ta @ git+ssh://git@github.com/Data-Analisis/Technical-Analysis-Indicators---Pandas.git',

    'lightweight-charts',
    'pynput',
    'pandas-market-calendars==4.5',
    'pytz',
]

platform_deps = [
    'PyGObject==3.50.0; platform_system=="Linux"',
    'pyqt; platform_system=="Linux"',
    'pywebview[qt]; platform_system=="Linux"',
    'pywebview; platform_system=="Darwin"',
]

setup(
    name='gym_trade',
    version='1.0',
    python_requires='>=3.9,<3.14',
    install_requires=install_requires + platform_deps,
    packages=find_packages(),
)
