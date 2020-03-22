#!/usr/bin/env python

import platform
import time
import numpy
from pyscf import lib
from pyscf.dft import uks

from mpi4pyscf.lib import logger
from mpi4pyscf.scf import uhf as mpi_uhf
from mpi4pyscf.dft import rks as mpi_rks
from mpi4pyscf.tools import mpi

comm = mpi.comm
rank = mpi.rank


@mpi.parallel_call(skip_args=[1, 2, 3, 4], skip_kwargs=['dm_last', 'vhf_last'])
def get_veff(mf, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    t0 = (time.clock(), time.time())
    mf.unpack_(comm.bcast(mf.pack()))
    mol = mf.mol
    ni = mf._numint

    if mf.nlc != '':
        raise NotImplementedError
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)

    # Broadcast the large input arrays here.
    if any(comm.allgather(isinstance(dm, str) and dm == 'SKIPPED_ARG')):
        if rank == 0 and dm is None:
            dm = mf.make_rdm1()
        dm = mpi.bcast_tagged_array(dm)

    if any(comm.allgather(isinstance(dm_last, str) and dm_last == 'SKIPPED_ARG')):
        dm_last = mpi.bcast_tagged_array(dm_last)

    if any(comm.allgather(isinstance(vhf_last, str) and vhf_last == 'SKIPPED_ARG')):
        vhf_last = mpi.bcast_tagged_array(vhf_last)

    ground_state = (dm.ndim == 3 and dm.shape[0] == 2)

    if mf.grids.coords is None:
        mpi_rks._setup_grids_(mf, dm[0]+dm[1])
        t0 = logger.timer(mf, 'setting up grids', *t0)

    if hermi == 2:  # because rho = 0
        n, exc, vxc = 0, 0, 0
    else:
        n, exc, vxc = ni.nr_uks(mol, mf.grids, mf.xc, dm)
        n = comm.allreduce(n)
        exc = comm.allreduce(exc)
        vxc = mpi.reduce(vxc)
        logger.debug(mf, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(mf, 'vxc', *t0)

    if abs(hyb) < 1e-10 and abs(alpha) < 1e-10:
        vk = None
        if getattr(vhf_last, 'vj', None) is not None:
            ddm = numpy.asarray(dm) - dm_last
            ddm = ddm[0] + ddm[1]
            vj = mpi.reduce(mf.get_j(mol, ddm, hermi))
            vj += vhf_last.vj
        else:
            vj = mf.get_j(mol, dm[0]+dm[1], hermi)
            vj = mpi.reduce(vj)
        vxc += vj
    else:
        if getattr(vhf_last, 'vk', None) is not None:
            ddm = numpy.asarray(dm) - dm_last
            vj, vk = mf.get_jk(mol, ddm, hermi)
            vj = mpi.reduce(vj[0] + vj[1])
            vk = mpi.reduce(vk) * hyb
            if abs(omega) > 1e-10:
                vklr = mf.get_k(mol, ddm, hermi, omega=omega)
                vklr = mpi.reduce(vklr) * (alpha - hyb)
                vk += vklr
            ddm = None
            vj += vhf_last.vj
            vk += vhf_last.vk
        else:
            vj, vk = mf.get_jk(mol, dm, hermi)
            vj = mpi.reduce(vj[0] + vj[1])
            vk = mpi.reduce(vk) * hyb
            if abs(omega) > 1e-10:
                vklr = mf.get_k(mol, dm, hermi, omega=omega)
                vklr = mpi.reduce(vklr) * (alpha - hyb)
                vk += vklr
        vxc += vj
        vxc -= vk

        if ground_state:
            exc -=(numpy.einsum('ij,ji', dm[0], vk[0]) +
                   numpy.einsum('ij,ji', dm[1], vk[1])) * .5

    if ground_state:
        ecoul = numpy.einsum('ij,ji', dm[0]+dm[1], vj) * .5
    else:
        ecoul = None

    vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)
    return vxc


@mpi.register_class
class UKS(uks.UKS, mpi_uhf.UHF):

    get_jk = mpi_uhf.UHF.get_jk
    get_j = mpi_uhf.UHF.get_j
    get_k = mpi_uhf.UHF.get_k

    @lib.with_doc(uks.UKS.get_veff.__doc__)
    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        assert(mol is None or mol is self.mol)
        return get_veff(self, None, dm, dm_last, vhf_last, hermi)

    def pack(self):
        return {'verbose': self.verbose,
                'direct_scf_tol': self.direct_scf_tol,
                'xc': self.xc,
                'nlc': self.nlc,
                'omega': self.omega,
                'small_rho_cutoff': self.small_rho_cutoff, }

    def dump_flags(self, verbose=None):
        mpi_info = mpi.platform_info()
        if rank == 0:
            uks.UKS.dump_flags(self, verbose)
            lib.logger.debug(self, 'MPI info (rank, host, pid)  %s', mpi_info)
        return self

