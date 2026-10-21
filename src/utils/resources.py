"""
resources.py
------------
Compute resources of the running job, for scripts submitted through SLURM.

    from resources import n_cpus, report_usage
    model = Model(n_jobs=n_cpus())
    ...
    report_usage()          # at the end: elapsed time and peak memory, printed to the log

n_cpus() counts the CPUs the job may actually use (its SLURM allocation), not all CPUs
of the node, so parallel code does not take cores that belong to other users' jobs.
"""

import os
import resource
import time

_START = time.monotonic()


def n_cpus():
    """Number of CPUs available to this process."""
    return len(os.sched_getaffinity(0))


def peak_memory_gb():
    """Peak memory (resident set) of this process so far, in GB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2   # kB on Linux


def report_usage():
    """Prints the elapsed time since import and the peak memory of this process."""
    elapsed = time.monotonic() - _START
    h, rem = divmod(int(elapsed), 3600)
    print(f"\nResources: {h:d}:{rem // 60:02d}:{rem % 60:02d} elapsed, "
          f"peak memory {peak_memory_gb():.1f} GB, {n_cpus()} CPUs available")
