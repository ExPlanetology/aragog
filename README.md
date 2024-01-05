# SPIDER
**Simulating Planetary Interior Dynamics with Extreme Rheology**

This is a pure Python version of the [SPIDER code](https://github.com/djbower/spider). This version does not support quadruple precision and applies conventional finite volumes to solve the system, i.e., the auxilliary variable approach outlined in Bower et al., 2018 is not invoked. Nevertheless, this pure Python version will probably prove to be more convenient for future development, particularly given that the atmosphere module is now supported by a separate (and more comprehensive) Python package *atmodeller*.

See [this setup guide](https://gist.github.com/djbower/c66474000029730ac9f8b73b96071db3) for setting up your system to develop *pySPIDER*.

## 2. References

#### 1. SPIDER code (interior dynamics)
Bower, D.J., P. Sanan, and A.S. Wolf (2018), Numerical solution of a non-linear conservation law applicable to the interior dynamics of partially molten planets, Phys. Earth Planet. Inter., 274, 49-62, doi: 10.1016/j.pepi.2017.11.004, arXiv: <https://arxiv.org/abs/1711.07303>, EarthArXiv: <https://eartharxiv.org/k6tgf>

#### 2. MgSiO3 melt data tables (RTpress) within SPIDER
Wolf, A.S. and D.J. Bower (2018), An equation of state for high pressure-temperature liquids (RTpress) with application to MgSiO3 melt, Phys. Earth Planet. Inter., 278, 59-74, doi: 10.1016/j.pepi.2018.02.004, EarthArXiv: <https://eartharxiv.org/4c2s5>

#### 3. Volatile and atmosphere coupling
Bower, D.J., Kitzmann, D., Wolf, A.S., Sanan, P., Dorn, C., and Oza, A.V. (2019), Linking the evolution of terrestrial interiors and an early outgassed atmosphere to astrophysical observations, Astron. Astrophys., 631, A103, doi: 10.1051/0004-6361/201935710, arXiv: <https://arxiv.org/abs/1904.08300>

#### 4. Redox reactions
Bower, D.J., Hakim, K., Sossi, P.A., and Sanan, P. (2022), Retention of water in terrestrial magma oceans and carbon-rich early atmospheres, Planet. Sci. J., 3, 93, doi: 10.3847/PSJ/ac5fb1, arXiv: <https://arxiv.org/abs/2110.08029>
