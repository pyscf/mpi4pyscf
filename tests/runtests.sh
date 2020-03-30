#!/bin/bash

# mpiexec -np 2 runtests.sh

# https://www.python.org/dev/peps/pep-0565/
#export PYTHONWARNINGS=once # Warn once per Python process
#export PYTHONWARNINGS=ignore # Never warn

if [ -z $OMPI_COMM_WORLD_RANK ] || [ $OMPI_COMM_WORLD_RANK == 0 ]; then
  pytest --cov-config=.coveragerc --cov-branch \
    --cov-report=term-missing --cov=mpi4pyscf --showlocals tests
else
  python -c 'import mpi4pyscf'
fi
