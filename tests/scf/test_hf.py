#!/usr/bin/env python

import pytest
import numpy
from pyscf import gto, scf
from mpi4pyscf import scf as mpi_scf

@pytest.fixture
def get_mol(scope='module'):
    mol = gto.M(atom='H 0 -.5 0; H 0 .5 4; H -1.1 0.2 0.2; H 0.6 0.5 0.4',
                basis='cc-pvdz')
    return mol


def test_jk(get_mol):
    mol = get_mol
    nao = mol.nao
    numpy.random.seed(1)
    dm = numpy.random.random((2,nao,nao))
    mf = mpi_scf.RHF(mol)

    vj, vk = mf.get_jk(mol, dm, hermi=0)
    vj0, vk0 = scf.hf.get_jk(mol, dm, hermi=0)
    assert abs(vj0-vj).max() < 1e-9
    assert abs(vk0-vk).max() < 1e-9
    vj = mf.get_j(mol, dm, hermi=0)
    vk = mf.get_k(mol, dm, hermi=0)
    assert abs(vj0-vj).max() < 1e-9
    assert abs(vk0-vk).max() < 1e-9

    dm = dm + dm.transpose(0,2,1)
    vj, vk = mpi_scf.hf.get_jk(mol, dm, hermi=1)
    vj0, vk0 = scf.hf.get_jk(mol, dm)
    assert abs(vj0-vj).max() < 1e-9
    assert abs(vk0-vk).max() < 1e-9
    vj = mpi_scf.hf.get_j(mol, dm, hermi=1)
    vk = mpi_scf.hf.get_k(mol, dm, hermi=1)
    assert abs(vj0-vj).max() < 1e-9
    assert abs(vk0-vk).max() < 1e-9

def test_mpi_uhf(get_mol):
    mol = get_mol
    mf = mpi_scf.UHF(mol)
    mf.direct_scf_tol = 1e-9
    mf.kernel()
    assert abs(mf.e_tot - -1.8562369268171945) < 1e-9

