Installation
============

Quick install
-------------

The basic procedure is to install Aragog into a Python environment. For example, if you are using a Conda distribution to create and manage Python environments (e.g. `Anaconda <https://www.anaconda.com/download>`_), create a new environment noting that Aragog requires Python >= 3.10. Once created, make sure to activate the environment. To achieve this, terminal commands are given below, but you can also use the Anaconda Navigator (or similar GUI) to create and activate environments:

.. code-block:: shell

    conda create -n aragog python
    conda activate aragog

Alternatively, you can create and activate a `virtual environment <https://docs.python.org/3/library/venv.html>`_.

Finally, install Aragog into the activated environment:

.. code-block:: shell

    pip install aragog

Developer install
-----------------

Navigate to a location on your computer and obtain the source code using git:

.. code-block:: shell

    git clone git@github.com:ExPlanetology/aragog.git aragog
    cd aragog

Install Aragog into the environment using either (a) `Poetry <https://python-poetry.org>`_ or (b) `pip <https://pip.pypa.io/en/stable/getting-started/>`_. There are some subtle differences between Poetry and pip, but in general Aragog is configured to be interoperable for most common operations (e.g. see this `Gist <https://gist.github.com/djbower/e9538e7eb5ed3deaf3c4de9dea41ebcd>`_).

(a) Poetry option, which requires that `Poetry <https://python-poetry.org>`_ is installed:

    .. code-block:: shell

        poetry install --all-extras

(b) pip option, where the ``-e`` option is for an `editable install <https://setuptools.pypa.io/en/latest/userguide/development_mode.html>`_:

    .. code-block:: shell

        pip install -e ".[docs]"

    If desired, you will need to manually install the dependencies for the tests, which are automatically installed by Poetry but not by pip. See the additional dependencies to install in `pyproject.toml`.

More comprehensive set up guides are available here:

- `VS Code and Poetry guide <https://gist.github.com/djbower/c66474000029730ac9f8b73b96071db3>`_
- `Windows and Spyder guide <https://gist.github.com/djbower/c82b4a70a3c3c74ad26dc572edefdd34>`_