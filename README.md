# SPIDER
**Simulating Planetary Interior Dynamics with Extreme Rheology**

## 1. Quick start

Formally you don't have to use VSCode or Poetry, but using them makes it easier to develop *SPIDER* as a community. If you are a Windows or Linux user (or use a different IDE such as Spyder), please send me your installation instructions so I can update this README.

1. Install [VSCode](https://code.visualstudio.com) if you don't already have it.
1. In VSCode you are recommended to install the following extensions:
	- Black Formatter
	- Code Spell Checker
 	- IntelliCode
	- isort
	- Jupyter
	- Pylance
	- Pylint
	- Region Viewer
	- Todo Tree
1. Install [Poetry](https://python-poetry.org) if you don't already have it.
1. Clone this repository (*spider*) to a local directory
1. In VSCode, go to *File* and *Open Folder...* and select the *spider* directory
1. We want to set up a virtual Python environment in the root directory of *spider*. An advantage of using a virtual environment is that it remains completely isolated from any other Python environments on your system (e.g. Conda or otherwise). You must have a Python interpreter available to build the virtual environment according to the dependency in `pyproject.toml`, which could be a native version on your machine or a version from a Conda environment that is currently active. You only need a Python binary so it is not required to install any packages. You can create a virtual environment by using the terminal in VSCode, where you may need to update `python` to reflect the location of the Python binary file. This will create a local Python environment in the `.venv` directory:
	
    ```
    python -m venv .venv
    ```
1. Open a new terminal window in VSCode and VSCode should recognise that you have a virtual environment in .venv, and load this environment automatically. You should see `(.venv)` as the prefix in the terminal prompt.
1. Install the project into the virtual environment using poetry to install all the required Python package dependencies:

    ```
    poetry install
    ```

To ensure that all developers are using the same settings for linting and formatting (e.g., using pylint, black, isort, as installed as extensions in step 2) there is a `settings.json` file in the `.vscode` directory. These settings will take precedence over your user settings for this project only.


## 2. References

#### 1. SPIDER code (interior dynamics)
Bower, D.J., P. Sanan, and A.S. Wolf (2018), Numerical solution of a non-linear conservation law applicable to the interior dynamics of partially molten planets, Phys. Earth Planet. Inter., 274, 49-62, doi: 10.1016/j.pepi.2017.11.004, arXiv: <https://arxiv.org/abs/1711.07303>, EarthArXiv: <https://eartharxiv.org/k6tgf>

#### 2. MgSiO3 melt data tables (RTpress) within SPIDER
Wolf, A.S. and D.J. Bower (2018), An equation of state for high pressure-temperature liquids (RTpress) with application to MgSiO3 melt, Phys. Earth Planet. Inter., 278, 59-74, doi: 10.1016/j.pepi.2018.02.004, EarthArXiv: <https://eartharxiv.org/4c2s5>

#### 3. Volatile and atmosphere coupling
Bower, D.J., Kitzmann, D., Wolf, A.S., Sanan, P., Dorn, C., and Oza, A.V. (2019), Linking the evolution of terrestrial interiors and an early outgassed atmosphere to astrophysical observations, Astron. Astrophys., 631, A103, doi: 10.1051/0004-6361/201935710, arXiv: <https://arxiv.org/abs/1904.08300>

#### 4. Redox reactions
Bower, D.J., Hakim, K., Sossi, P.A., and Sanan, P. (2022), Retention of water in terrestrial magma oceans and carbon-rich early atmospheres, Planet. Sci. J., 3, 93, doi: 10.3847/PSJ/ac5fb1, arXiv: <https://arxiv.org/abs/2110.08029>