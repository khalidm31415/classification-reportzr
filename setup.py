import os
from setuptools import setup

with open(os.path.join('classification_reportzr', 'VERSION')) as file:
    version = file.read().strip()

with open('requirements.txt') as file:
    requirements = file.read().splitlines()

with open('README.md') as file:
    long_description = file.read()

setup(
    name='classification-reportzr',
    version=version,
    description="Automate machine learning classification task report for Pak Zuherman",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/khalidm31415/classification-reportzr",
    keywords=['classification report', 'laporan klasifikasi', 'zuherman', 'zr'],
    packages=['classification_reportzr'],
    python_requires='>=3.6',
    zip_safe= False,
    include_package_data=True,
    package_data={'': ['VERSION']},
    install_requires=requirements
)
