#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import sys
import time

if sys.version_info < (3, 0):
    process_clock = time.clock
    perf_counter = time.time
else:
    process_clock = time.process_time
    perf_counter = time.perf_counter

from pyscf.lib import logger
from mpi4pyscf.tools import mpi
rank = mpi.rank

DEBUG4 = logger.DEBUG4
DEBUG3 = logger.DEBUG3
DEBUG2 = logger.DEBUG2
DEBUG1 = logger.DEBUG1
DEBUG  = logger.DEBUG
INFO   = logger.INFO
NOTE   = logger.NOTICE
WARNING = WARN = logger.WARN
ERROR  = ERR = logger.ERR
QUIET  = logger.QUIET
CRIT   = logger.CRIT
ALERT  = logger.ALERT
PANIC  = logger.PANIC

TIMER_LEVEL  = logger.TIMER_LEVEL

log = logger.log
error = logger.error
warn = logger.warn
note = logger.note
info = logger.info
debug  = logger.debug
debug1 = logger.debug1
debug2 = logger.debug2
debug3 = logger.debug3
debug4 = logger.debug4
timer = logger.timer
timer_debug1 = logger.timer_debug1


if rank > 0:
    def flush(rec, msg, *args):
        pass
    logger.flush = flush

    def allflush(rec, msg, *args):
        sys.stdout.write('[rank %d] ' % rank)
        sys.stdout.write(msg%args)
        sys.stdout.write('\n')
        sys.stdout.flush()
else:
    def allflush(rec, msg, *args):
        rec.stdout.write('[rank %d] ' % rank)
        rec.stdout.write(msg%args)
        rec.stdout.write('\n')
        rec.stdout.flush()

def alllog(rec, msg, *args):
    if rec.verbose > logger.QUIET:
        allflush(rec, msg, *args)

def allerror(rec, msg, *args):
    if rec.verbose >= logger.ERROR:
        allflush(rec, 'Error: '+msg, *args)
    sys.stderr.write('[rank %d] Error: '%rank + (msg%args) + '\n')

def allwarn(rec, msg, *args):
    if rec.verbose >= logger.WARN:
        allflush(rec, 'Warn: '+msg, *args)
        if rec.stdout is not sys.stdout:
            sys.stderr.write('[rank %d] Warn: '%rank + (msg%args) + '\n')

def allinfo(rec, msg, *args):
    if rec.verbose >= logger.INFO:
        allflush(rec, msg, *args)

def allnote(rec, msg, *args):
    if rec.verbose >= logger.NOTICE:
        allflush(rec, msg, *args)

def alldebug(rec, msg, *args):
    if rec.verbose >= logger.DEBUG:
        allflush(rec, msg, *args)

def alldebug1(rec, msg, *args):
    if rec.verbose >= logger.DEBUG1:
        allflush(rec, msg, *args)

def alldebug2(rec, msg, *args):
    if rec.verbose >= logger.DEBUG2:
        allflush(rec, msg, *args)

def alldebug3(rec, msg, *args):
    if rec.verbose >= logger.DEBUG3:
        allflush(rec, msg, *args)

def alldebug4(rec, msg, *args):
    if rec.verbose >= logger.DEBUG4:
        allflush(rec, msg, *args)

def allstdout(rec, msg, *args):
    if rec.verbose >= logger.DEBUG:
        allflush(rec, msg, *args)
    sys.stdout.write('[rank %d] >>> %s\n' % msg)

def alltimer(rec, msg, cpu0=None, wall0=None):
    if cpu0 is None:
        cpu0 = rec._t0
    if wall0:
        rec._t0, rec._w0 = process_clock(), perf_counter()
        if rec.verbose >= logger.TIMER_LEVEL:
            allflush(rec, '    CPU time for %s %9.2f sec, wall time %9.2f sec'
                     % (msg, rec._t0-cpu0, rec._w0-wall0))
        return rec._t0, rec._w0
    else:
        rec._t0 = time.clock()
        if rec.verbose >= logger.TIMER_LEVEL:
            allflush(rec, '    CPU time for %s %9.2f sec' % (rec._t0-cpu0))
        return rec._t0

def alltimer_debug1(rec, msg, cpu0=None, wall0=None):
    if rec.verbose >= logger.DEBUG1:
        return alltimer(rec, msg, cpu0, wall0)
    elif wall0:
        rec._t0, rec._w0 = process_clock(), perf_counter()
        return rec._t0, rec._w0
    else:
        rec._t0 = time.clock()
        return rec._t0

def alltimer_debug2(rec, msg, cpu0=None, wall0=None):
    if rec.verbose >= logger.DEBUG2:
        return alltimer(rec, msg, cpu0, wall0)
    elif wall0:
        rec._t0, rec._w0 = process_clock(), perf_counter()
        return rec._t0, rec._w0
    else:
        rec._t0 = time.clock()
        return rec._t0


class Logger(logger.Logger):
    alllog = alllog
    allerror = allerror
    allwarn = allwarn
    allnote = allnote
    allinfo = allinfo
    alldebug  = alldebug
    alldebug1 = alldebug1
    alldebug2 = alldebug2
    alldebug3 = alldebug3
    alldebug4 = alldebug4
    alltimer = alltimer
    alltimer_debug1 = alltimer_debug1
    alltimer_debug2 = alltimer_debug2

def new_logger(rec=None, verbose=None):
    if isinstance(verbose, Logger):
        log = verbose
    elif isinstance(verbose, int):
        if hasattr(rec, 'stdout'):
            log = Logger(rec.stdout, verbose)
        else:
            log = Logger(sys.stdout, verbose)
    else:
        log = Logger(rec.stdout, rec.verbose)
    return log

