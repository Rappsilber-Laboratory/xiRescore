#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    "frosch~=0.1.9",
    "pandas~=2.0.1",
    "numpy~=1.24.3",
    "PyYAML~=6.0",
    "tabulate~=0.9.0",
    "setuptools~=65.5.1",
    "numpy-indexed~=0.3.7",
    "scikit-learn~=1.3.0",
    "joblib~=1.2.0",
    "xgboost~=1.7.5",
    "tqdm~=4.66.0",
    "hyperopt~=0.2.7",
    "lightgbm~=3.3.5",  # Check the pre-requirements https://pypi.org/project/lightgbm/
    "pyteomics~=4.6",
    "matplotlib~=3.7.1",
    "seaborn~=0.12.2",
    "imblearn~=0.0",
    "JPype1~=1.4.1",
    "XlsxWriter~=3.1.0",
    "pytest~=7.3.1",
    "pydocstyle~=6.3.0",
    "pytest-cov~=4.0.0",
    "pytest-flake8~=1.0.6",
    "pytest-pydocstyle~=2.3.2",
    "py~=1.11.0",
    "flake8==4.0.1",
    "networkx~=3.1",
    "pyspark~=3.5.0",
    "pyspark[sql]~=3.5.0",
    "multiprocess~=0.70.15",
    "deepmerge~=1.1.0",
    "SQLAlchemy~=2.0.30",
    "psycopg2>=2.9",
    "python-logging-loki",
    "fastparquet=2024.5.0",
    "pyarrow=16.0.0",
]

test_requirements = ['pytest>=3', ]

setup(
    author="Falk Boudewijn Schimweg",
    author_email='f.schimweg@win.tu-berlin.de',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.10',
    ],
    description="A machine learning based approach to rescore crosslinked spectrum matches (CSMs).",
    entry_points={
        'console_scripts': [
            'xirescore=xirescore.cli:main',
        ],
    },
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='xirescore',
    name='xirescore',
    packages=find_packages(include=['xirescore', 'xirescore.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/z3rone/xirescore',
    version='0.1.0',
    zip_safe=False,
)
