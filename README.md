# Aragog

[![Documentation Status](https://readthedocs.org/projects/aragog/badge/?version=latest)](https://aragog.readthedocs.io/en/latest/?badge=latest)

## Under development

This code remains under active development, hence the interface is not stable and should not be relied upon.

## About

Aragog is a Python package that computes the 1-D interior dynamics of rocky mantles that are solid, liquid, or mixed phase. It is mostly a pure Python version of the [SPIDER code](https://github.com/djbower/spider) originally written in C albeit with some differences. Note that the atmosphere module in the original SPIDER code is now supported by a separate and more comprehensive Python package Atmodeller (release forthcoming).

Documentation: <https://aragog.readthedocs.io>

Source code: <https://github.com/ExPlanetology/aragog>

## Citation

If you use Aragog (or for that matter the original [SPIDER code](https://github.com/djbower/spider)) please cite:

- Bower, D.J., P. Sanan, and A.S. Wolf (2018), Numerical solution of a non-linear conservation law applicable to the interior dynamics of partially molten planets, Phys. Earth Planet. Inter., 274, 49-62, doi: <https://doi.org/10.1016/j.pepi.2017.11.004>, arXiv: <https://arxiv.org/abs/1711.07303>, EarthArXiv: <https://eartharxiv.org/k6tgf>

## Installation

### Quick install

The basic procedure is to install Aragog into a Python environment. For example, if you are using a Conda distribution to create and manage Python environments (e.g. [Anaconda](https://www.anaconda.com/download)), create a new environment noting that Aragog requires Python >= 3.10. Once created, make sure to activate the environment. To achieve this, terminal commands are given below, but you can also use the Anaconda Navigator (or similar GUI) to create and activate environments:

    conda create -n aragog python
    conda activate aragog

Alternatively, you can create and activate a [virtual environment](https://docs.python.org/3/library/venv.html).

Finally, install Aragog into the activated environment:

	pip install aragog

### Developer install

> - See this [guide](https://gist.github.com/djbower/c66474000029730ac9f8b73b96071db3) to develop Aragog using [VS Code](https://code.visualstudio.com) and [Poetry](https://python-poetry.org).
> - See this [guide](https://gist.github.com/djbower/c82b4a70a3c3c74ad26dc572edefdd34) to develop Aragog if you are a Windows or Spyder user.

Navigate to a location on your computer and obtain the source code using git:

    git clone git@github.com:ExPlanetology/aragog.git aragog
    cd aragog

Install Aragog into the environment using either (a) [Poetry](https://python-poetry.org) or (b) [pip](https://pip.pypa.io/en/stable/getting-started/). There are some subtle differences between Poetry and pip, but in general Aragog is configured to be interoperable for most common operations (e.g. see this [Gist](https://gist.github.com/djbower/e9538e7eb5ed3deaf3c4de9dea41ebcd)).

- (a) Poetry option, which requires that [Poetry](https://python-poetry.org) is installed:

		poetry install --all-extras

- (b) pip option, where the `-e` option is for an [editable install](https://setuptools.pypa.io/en/latest/userguide/development_mode.html):

		pip install -e ".[docs]"

	If desired, you will need to manually install the dependencies for the tests, which are automatically installed by Poetry but not by `pip`. See the additional dependencies to install in `pyproject.toml`.


## Other references

- Wolf, A.S. and D.J. Bower (2018), An equation of state for high pressure-temperature liquids (RTpress) with application to MgSiO3 melt, Phys. Earth Planet. Inter., 278, 59-74, doi: 10.1016/j.pepi.2018.02.004, EarthArXiv: <https://eartharxiv.org/4c2s5>

- Bower, D.J., Kitzmann, D., Wolf, A.S., Sanan, P., Dorn, C., and Oza, A.V. (2019), Linking the evolution of terrestrial interiors and an early outgassed atmosphere to astrophysical observations, Astron. Astrophys., 631, A103, doi: 10.1051/0004-6361/201935710, arXiv: <https://arxiv.org/abs/1904.08300>

- Bower, D.J., Hakim, K., Sossi, P.A., and Sanan, P. (2022), Retention of water in terrestrial magma oceans and carbon-rich early atmospheres, Planet. Sci. J., 3, 93, doi: 10.3847/PSJ/ac5fb1, arXiv: <https://arxiv.org/abs/2110.08029>
