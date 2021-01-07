import os
from setuptools import setup, find_packages

with open(os.path.join('systementity', 'VERSION')) as file:
    version = file.read().strip()

with open('requirements.txt') as file:
    requirements = file.read().splitlines()

setup(
    name='classification-reportzr',
    version=version,
    description="Automate machine learning classification task report for Pak Zuherman",
    keywords=['classification report', 'laporan klasifikasi', 'zuherman', 'zr'],
    packages=find_packages(),
    python_requires='>=3.6',
    zip_safe= False,
    include_package_data=True,
    package_data={'': ['*.csv','*.json','*.pkl','*.txt', 'VERSION']},
    install_requires=requirements
)
