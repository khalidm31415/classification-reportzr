import os
from setuptools import setup

this_directory = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(this_directory, 'classification_reportzr', 'VERSION')) as file:
    version = file.read().strip()

with open(os.path.join(this_directory, 'requirements.txt')) as file:
    requirements = file.read().splitlines()

with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as file:
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
