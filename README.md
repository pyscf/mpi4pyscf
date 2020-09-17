An MPI plugin for PySCF
=======================

mpi4pyscf is a plugin for PySCF which enables MPI (Message Passing Interface) parallelism.

2020-09-16

* [Latest version 0.3.0](https://github.com/mpipyscf/mpipyscf/releases/tag/v0.3.0)

Quick start
-----------

When the script of the serial PySCF version works, MPI can be activated by
importing mpi4pyscf and replacing the corresponding initialization statements in
the script. For examples::
```
import pyscf
mol = pyscf..M(atom='''
O    0.   0.       0.
H    0.   -0.757   0.587
H    0.   0.757    0.587''',
               basis='cc-pvtz')

# Serial mode
from pyscf import scf
mf = scf.RHF(mol).run()

# MPI parallelism
from mpipyscf import scf
mf = scf.RHF(mol).run()
```

See more examples of usage in https://github.com/pyscf/mpi4pyscf/examples

Installation
------------

```
pip install mpi4pyscf
```

