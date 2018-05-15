#!/usr/bin/env python

import numpy
import scipy.linalg
from pyscf import lib

from mpi4pyscf.lib import logger
from mpi4pyscf.tools import mpi


class DistributedDIIS(lib.diis.DIIS):

    def extrapolate(self, nd=None):
        if nd is None:
            nd = self.get_num_vec()

        h = self._H[:nd+1,:nd+1].copy()
        h[1:,1:] = mpi.comm.allreduce(self._H[1:nd+1,1:nd+1])
        g = numpy.zeros(nd+1, h.dtype)
        g[0] = 1

        w, v = scipy.linalg.eigh(h)
        if numpy.any(abs(w)<1e-14):
            logger.debug(self, 'Singularity found in DIIS error vector space.')
            idx = abs(w)>1e-14
            c = numpy.dot(v[:,idx]*(1./w[idx]), numpy.dot(v[:,idx].T.conj(), g))
        else:
            try:
                c = numpy.linalg.solve(h, g)
            except numpy.linalg.linalg.LinAlgError as e:
                logger.warn(self, ' diis singular, eigh(h) %s', w)
                raise e
        logger.debug1(self, 'diis-c %s', c)

        xnew = None
        for i, ci in enumerate(c[1:]):
            xi = self.get_vec(i)
            if xnew is None:
                xnew = numpy.zeros(xi.size, c.dtype)
            for p0, p1 in lib.prange(0, xi.size, lib.diis.BLOCK_SIZE):
                xnew[p0:p1] += xi[p0:p1] * ci
        return xnew

