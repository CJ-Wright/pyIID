#Python Infinite Improbability Drive 
[![Build Status](https://magnum.travis-ci.com/CJ-Wright/pyIID.svg?token=5KvMJdLqpf5ZXrVzPA7y&branch=tringle_kernel)](https://magnum.travis-ci.com/CJ-Wright/pyIID)
[![Coverage Status](https://coveralls.io/repos/ZhouHUB/pyIID/badge.svg?branch=master&t=Kitk02)](https://coveralls.io/r/ZhouHUB/pyIID?branch=master)

Is designed for the Monte Carlo modeling of nanomaterials using atomic pair distributuion functions, other experimental data, and ab-initio structural calculations.

Areas that need improvement:

1. GPU based gradient and PDF potential energy functions: otherwise everything is very slow.
    1. Gradient: particularly expensive on single threaded systems, which makes Hamiltonian Monte Carlo *glacially* slow
    1. Generating PDFs: currently using [diffpy.srreal](https://github.com/diffpy/diffpy.srreal) for PDF generation, which is not GPU optimized
    1. PDF potential energy: this is quantified using the RW value which could also be put onto the GPU
2. Ab-initio calculation support:while most of this is handled by ASE issues include
    1. Scaling and Units for PDF comparison: we need a way to effectively tune the relationship between the PDF and the ab-initio, otherwise one will dominate the refinement and dynamics
    1. Support for non-ASE calculators