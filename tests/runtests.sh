#!/bin/bash

if [ -z $OMPI_COMM_WORLD_RANK ] || [ $OMPI_COMM_WORLD_RANK == 0 ]; then
  mpiexec -np ${NP:=2} pytest --cov-config=.coveragerc --cov-branch \
    --cov-report=term-missing --cov=mpi4pyscf --showlocals tests
else
  mpiexec -np ${NP:=2} python -c 'import mpi4pyscf'
fi
