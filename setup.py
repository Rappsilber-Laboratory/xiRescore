#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    "frosch~=0.1.9",
    "pandas>=2.0.0",
    "numpy>=1.7.2",
    "PyYAML~=6.0",
    "tabulate~=0.9.0",
    "numpy-indexed~=0.3.7",
    "scikit-learn>=1.3.2",
    "joblib~=1.2.0",
    "xgboost~=1.7.5",
    "tqdm~=4.66.0",
    "hyperopt~=0.2.7",
    "lightgbm~=3.3.5",  # Check the pre-requirements https://pypi.org/project/lightgbm/
    "imblearn~=0.0",
    "py~=1.11.0",
    "networkx~=3.1",
    "multiprocess~=0.70.16",
    "threadpoolctl>=3.5.0",
    "deepmerge~=1.1.0",
    "SQLAlchemy~=2.0.30",
    "psycopg2>=2.9",
    "python-logging-loki",
    "fastparquet>=2022.11.0",
    "pyarrow~=17.0.0",
    "scipy>=1.0.1",
    "setuptools~=71.1.0",
    "xiutilities==1.2.3"
]

requirements_dev = [
    'pip>=19.2.3',
    'bump2version>=0.5.11',
    'wheel>=0.33.6',
    'watchdog>=0.9.0',
    'flake8>=3.7.8',
    'coverage>=4.5.4',
    'Sphinx>=1.8.5',
    'twine>=1.14.0',
    'setuptools_scm',
    'pytest>=6.2.4',
    'black>=21.7b0',
    'sphinx_rtd_theme',
]

requirements_test = [
    'pytest>=3',
    "pydocstyle~=6.3.0",
    "pytest-cov~=4.0.0",
    "pytest-flake8~=1.0.6",
    "pytest-pydocstyle~=2.3.2",
    "flake8==4.0.1",
    "pytest-xdist",
]

requirements_docs = [
    "sphinx",
    "sphinx_rtd_theme",
    "sphinx-favicon",
]

setup(
    author="Falk Boudewijn Schimweg",
    author_email='git@falk.schimweg.de',
    python_requires='>=3.8',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.12',
    ],
    description="A machine learning based approach to rescore crosslinked spectrum matches (CSMs).",
    entry_points={
        'console_scripts': [
            'xirescore=xirescore.cli:main',
        ],
    },
    install_requires=requirements,
    extras_require={
        'test': requirements_test,
        'docs': requirements_docs,
        'dev': requirements_dev + requirements_test + requirements_docs,
    },
    license="GNU Affero General Public License v3 or later (AGPLv3+)",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='xirescore',
    name='xirescore',
    packages=find_packages(include=['xirescore', 'xirescore.*'], exclude=["tests"]),
    test_suite='tests',
    tests_require=requirements_test,
    url='https://github.com/Rappsilber-Laboratory/xiRescore',
    zip_safe=False,
)
