"""Must be the first import in any module that (directly or transitively,
e.g. via sentence_transformers/faiss/transformers) loads torch or faiss.

Diagnosed root cause of a reproducible Windows segfault (SIGSEGV, exit 139,
no Python traceback) when the retrieval index and stance classifier load in
the same process: faiss-cpu bundles its own OpenMP/BLAS runtime
(vcomp140.dll + libopenblas.dll) and torch bundles a separate one
(libiomp5md.dll, MKL-backed). Both size their internal thread pools to the
CPU core count at native-library-load time. Running both multithreaded in
one process crashes on first concurrent use.

Confirmed experimentally (not guessed):
- RSS at crash time was ~480MB on a 16GB machine — this is a threading bug,
  not memory pressure, despite superficially resembling one.
- Calling torch.set_num_threads(1) / faiss.omp_set_num_threads(1) AFTER
  import does NOT prevent the crash — these libraries read
  OMP_NUM_THREADS/MKL_NUM_THREADS/OPENBLAS_NUM_THREADS at DLL-load time, so
  the env vars must be set before torch/faiss are first imported anywhere
  in the process.
- KMP_DUPLICATE_LIB_OK=TRUE alone does not fix it (this isn't a duplicate-
  symbol abort, it's a concurrent-thread-pool crash).

Forcing single-threaded BLAS/OpenMP has no meaningful cost here: passages
are scored one at a time and the corpus is a few hundred passages, so
intra-op parallelism was never buying real throughput.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
