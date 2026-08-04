"""Pins BLAS/OpenMP to a single thread.

faiss-cpu and torch each bundle their own OpenMP runtime and size their thread
pools at DLL-load time, which segfaults on Windows when both run multithreaded
in one process. These variables are only read at load time, so this module must
be imported before torch or faiss anywhere in the process.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
