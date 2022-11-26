# ScandiNLI

Natural language inference for the Scandinavian languages

______________________________________________________________________
[![Documentation](https://img.shields.io/badge/docs-passing-green)](https://saattrupdan.github.io/scandinli/scandinli.html)
[![License](https://img.shields.io/github/license/saattrupdan/scandinli)](https://github.com/saattrupdan/scandinli/blob/main/LICENSE)
[![LastCommit](https://img.shields.io/github/last-commit/saattrupdan/scandinli)](https://github.com/saattrupdan/scandinli/commits/main)
[![Code Coverage](https://img.shields.io/badge/Coverage-0%25-red.svg)](https://github.com/saattrupdan/scandinli/tree/main/tests)


Developers:

- Dan Saattrup Nielsen (dan.nielsen@alexandra.dk)


## Setup

### Set up the environment

1. Run `make install`, which installs Poetry (if it isn't already installed), sets up a virtual environment and all Python dependencies therein.
2. Run `source .venv/bin/activate` to activate the virtual environment.

### Install new packages

To install new PyPI packages, run:

```
$ poetry add <package-name>
```

### Auto-generate API documentation

To auto-generate API document for your project, run:

```
$ make docs
```

To view the documentation, run:

```
$ make view-docs
```

## Tools used in this project
* [Poetry](https://towardsdatascience.com/how-to-effortlessly-publish-your-python-package-to-pypi-using-poetry-44b305362f9f): Dependency management
* [hydra](https://hydra.cc/): Manage configuration files
* [pre-commit plugins](https://pre-commit.com/): Automate code reviewing formatting
* [pdoc](https://github.com/pdoc3/pdoc): Automatically create an API documentation for your project

## Project structure
```
.
├── .flake8
├── .github
│   └── workflows
│       ├── ci.yaml
│       └── docs.yaml
├── .gitignore
├── .pre-commit-config.yaml
├── CHANGELOG.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── config
│   ├── __init__.py
│   ├── config.yaml
│   ├── model
│   │   └── model1.yaml
│   └── process
│       └── process1.yaml
├── data
│   ├── final
│   ├── processed
│   └── raw
├── makefile
├── models
├── notebooks
├── poetry.toml
├── pyproject.toml
├── src
│   ├── scripts
│   │   ├── fix_dot_env_file.py
│   │   └── versioning.py
│   └── scandinli
│       ├── __init__.py
│       └── demo.py
└── tests
    └── __init__.py
```
