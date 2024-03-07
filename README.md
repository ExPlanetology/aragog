# SPIDER
**Simulating Planetary Interior Dynamics with Extreme Rheology**

## About

This is a pure Python version of the [SPIDER code](https://github.com/djbower/spider). This version does not support quadruple precision and applies conventional finite volumes to solve the system, i.e., the auxilliary variable approach outlined in Bower et al., 2018 is not invoked. Nevertheless, this pure Python version will probably prove to be more convenient for future development, particularly given that the atmosphere module is now supported by a separate and more comprehensive Python package *Atmodeller*.

## Citation

### 1. SPIDER code (interior dynamics)
Bower, D.J., P. Sanan, and A.S. Wolf (2018), Numerical solution of a non-linear conservation law applicable to the interior dynamics of partially molten planets, Phys. Earth Planet. Inter., 274, 49-62, doi: 10.1016/j.pepi.2017.11.004, arXiv: <https://arxiv.org/abs/1711.07303>, EarthArXiv: <https://eartharxiv.org/k6tgf>

### 2. MgSiO3 melt data tables (RTpress) within SPIDER
Wolf, A.S. and D.J. Bower (2018), An equation of state for high pressure-temperature liquids (RTpress) with application to MgSiO3 melt, Phys. Earth Planet. Inter., 278, 59-74, doi: 10.1016/j.pepi.2018.02.004, EarthArXiv: <https://eartharxiv.org/4c2s5>

### 3. Volatile and atmosphere coupling
Bower, D.J., Kitzmann, D., Wolf, A.S., Sanan, P., Dorn, C., and Oza, A.V. (2019), Linking the evolution of terrestrial interiors and an early outgassed atmosphere to astrophysical observations, Astron. Astrophys., 631, A103, doi: 10.1051/0004-6361/201935710, arXiv: <https://arxiv.org/abs/1904.08300>

### 4. Redox reactions
Bower, D.J., Hakim, K., Sossi, P.A., and Sanan, P. (2022), Retention of water in terrestrial magma oceans and carbon-rich early atmospheres, Planet. Sci. J., 3, 93, doi: 10.3847/PSJ/ac5fb1, arXiv: <https://arxiv.org/abs/2110.08029>

## Installation

*Spider* is a Python package that can be installed on a variety of platforms (e.g. Mac, Windows, Linux).

### Quick install

If you want a GUI way of installing *Spider*, particularly if you are a Windows or Spyder user, see [here](https://gist.github.com/djbower/c82b4a70a3c3c74ad26dc572edefdd34). Otherwise, the instructions below should work to install *Spider* using the terminal on a Mac or Linux system.

Navigate to a location on your computer and obtain the *Spider* source code:

    git clone git@github.com:ExPlanetology/pyspider.git spider
    cd spider

The basic procedure is to install *Spider* into an environment. For example, if you are using a Conda distribution to create Python environments (e.g. [Anaconda](https://www.anaconda.com/download)), create a new environment to install *Spider*. *Spider* requires Python >= 3.10:

    conda create -n spider python
    conda activate spider

Install *Spider* into the environment. The preference is to use [Poetry](https://python-poetry.org) because it allows greater flexibility and control over dependency management, and this is actually required if you want to install the dependencies for testing and documentation that are unfortunately not yet supported by `pip`. However, you can install the main *Atmodeller* package using pip as follows, where you can include the `-e` option if you want an [editable install ](https://setuptools.pypa.io/en/latest/userguide/development_mode.html).

    pip install .

### Developer install

See this [developer setup guide](https://gist.github.com/djbower/c66474000029730ac9f8b73b96071db3) to set up your system to develop *Spider* using [VS Code](https://code.visualstudio.com) and [Poetry](https://python-poetry.org).

## Documentation

Documentation will eventually be available on readthedocs, but for the time being you can compile (and contribute if you wish) to the documentation in the `docs/` directory. To compile the documentation you will need to use Poetry and the option `--with docs` when you run `poetry install`. See [here](https://python-poetry.org/docs/managing-dependencies/) for further information.

## Tests

You can confirm that all tests pass by running `pytest` in the root directory of *Spider*. Please add more tests if you add new features. Note that `pip install .` in the *Quick install* instructions will not install `pytest` so you will need to install `pytest` into the environment separately.
