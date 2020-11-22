from setuptools import setup

setup(
    name='tensorseason',
    version='0.1',
    description='Seasons as tensors',
    author='Caner Turkmen, Melih Barsbey',
    author_email='caner.turkmen@boun.edu.tr',
    license='MIT',
    package_dir={"": "src"},
    packages=['tensorseason'],
    install_requires=[
        "gluonts",
        "pandas>=1",
        "numpy>=1.19",
        "scipy>=1.3",
        "tensorly>=0.5",
        "tqdm>=4",
    ]
)

