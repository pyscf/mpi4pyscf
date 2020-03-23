#!/usr/bin/env python

import pytest
import numpy
from pyscf import gto, dft
from mpi4pyscf import dft as mpi_dft

@pytest.fixture
def get_mol(scope='module'):
    mol = gto.M(atom='''
O 0 0     0
H 0 -.757 .587
H 0  .757 .587''',
                basis='cc-pvdz')
    return mol


def test_veff(get_mol):
    mol = get_mol
    nao = mol.nao
    numpy.random.seed(1)
    dm = numpy.random.random((2,nao,nao))
    dm = dm + dm.transpose(0,2,1)

    mf = mpi_dft.RKS(mol)
    mf.xc = 'b3lyp'
    vxc = mf.get_veff(mol, dm)
    mf0 = mol.RKS(mol, xc='b3lyp')
    vxc0 = mf0.get_veff(mol, dm)
    print(abs(vxc0-vxc).max())

    mol1 = mol.copy()
    mol1.spin = 2
    mf = mpi_dft.UKS(mol1)
    mf.xc = 'b3lyp'
    vxc = mf.get_veff(mol1, dm)
    mf0 = mol1.UKS(mol1, xc='b3lyp')
    vxc0 = mf0.get_veff(mol1, dm)
    assert abs(vxc0-vxc).max() < 1e-9

def test_mpi_uks(get_mol):
    mol = get_mol
    mf = mpi_dft.UKS(mol)
    mf.xc = 'b3lyp'
    mf.direct_scf_tol = 1e-9
    mf.kernel()
    eref = mol.UKS(xc='b3lyp').run().e_tot
    assert abs(mf.e_tot - -76.38322442598239) < 1e-9
    assert abs(mf.e_tot - eref) < 1e-9

