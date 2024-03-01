Installation
============

*Spider* is a Python package that can be installed on a variety of platforms (e.g. Mac, Windows, Linux).

Quick install
-------------

The instructions are given in terms of terminal commands for a Mac, but equivalents exist for other systems.

Navigate to a location on your computer and obtain the *Spider* source code::

    git clone git@github.com:ExPlanetology/pyspider.git spider
    cd spider

The basic procedure is to install *Spider* into an environment. For example, if you are using a Conda distribution to create Python environments (e.g. `Anaconda <https://www.anaconda.com/download>`_), create a new environment to install *Spider*. *Spider* requires Python >= 3.10::

    conda create -n spider python
    conda activate spider

Install spider into the environment, where you can include the ``-e`` option if you want an `editable install <https://setuptools.pypa.io/en/latest/userguide/development_mode.html>`_::

    pip install .

Developer install
-----------------

See this `developer setup guide <https://gist.github.com/djbower/c66474000029730ac9f8b73b96071db3>`_ to set up your system to develop *Spider* using `VS Code <https://code.visualstudio.com>`_ and `Poetry <https://python-poetry.org>`_.

Tutorial
--------

Several Jupyter notebook tutorials are provided in `notebooks/`.

Tests
-----

You can confirm that all tests pass by running ``pytest`` in the root directory of *Spider*. Please add more tests if you add new features. Note that ``pip install .`` in the *Quick install* instructions will not install ``pytest`` so you will need to install ``pytest`` separately.