from chainer.backends import cuda
import numpy

try:
    import cupy
    _cupy_available = True
except ImportError:
    _cupy_available = False

if _cupy_available:
    from cupy.cuda import cublas
    from cupy.cuda import cusolver
    from cupy.cuda import device


def inverse(a, use_cholesky=True):

    xp = cuda.get_array_module(a)
    if xp == numpy:
        if use_cholesky:
            la = numpy.linalg.cholesky(a)
            la_inv = numpy.linalg.inv(la)
            return numpy.matmul(la_inv.T, la_inv)
        else:
            return numpy.linalg.inv(a)

    if not _cupy_available:
        raise RuntimeError('CuPy is not available')
    if not cupy.cuda.cusolver_enabled:
        raise RuntimeError('cuSolver is not available')

    cupy.linalg.util._assert_cupy_array(a)
    cupy.linalg.util._assert_rank2(a)
    cupy.linalg.util._assert_nd_squareness(a)

    if a.dtype.char == 'f' or a.dtype.char == 'd':
        dtype = a.dtype.char
    else:
        dtype = numpy.find_common_type((a.dtype.char, 'f'), ()).char

    cusolver_handle = device.get_cusolver_handle()
    dev_info = cupy.empty(1, dtype=cupy.int)
    m = a.shape[0]

    b = cupy.eye(m, dtype=dtype)

    if not use_cholesky:
        if dtype == 'f':
            getrf = cusolver.sgetrf
            getrf_bufferSize = cusolver.sgetrf_bufferSize
            getrs = cusolver.sgetrs
        else:  # dtype == 'd'
            getrf = cusolver.dgetrf
            getrf_bufferSize = cusolver.dgetrf_bufferSize
            getrs = cusolver.dgetrs

        buffersize = getrf_bufferSize(cusolver_handle, m, m, a.data.ptr, m)

        # TODO(y1r): cache buffer to avoid malloc
        workspace = cupy.empty(buffersize, dtype=dtype)
        ipiv = cupy.empty((a.shape[0], 1), dtype=dtype)

        # LU Decomposition
        getrf(cusolver_handle, m, m, a.data.ptr, m,
              workspace.data.ptr, ipiv.data.ptr, dev_info.data.ptr)

        # TODO(y1r): check dev_info status

        # solve for the inverse
        getrs(cusolver_handle, 0, m, m, a.data.ptr, m,
              ipiv.data.ptr, b.data.ptr, m, dev_info.data.ptr)

        # TODO(y1r): check dev_info status
    else:
        if dtype == 'f':
            potrf = cusolver.spotrf
            potrf_bufferSize = cusolver.spotrf_bufferSize
            potrs = cusolver.spotrs
        else:  # dtype == 'd'
            potrf = cusolver.dpotrf
            potrf_bufferSize = cusolver.dpotrf_bufferSize
            potrs = cusolver.dpotrs

        buffersize = potrf_bufferSize(
            cusolver_handle, cublas.CUBLAS_FILL_MODE_UPPER, m, a.data.ptr, m)

        # TODO(y1r): cache buffer to avoid malloc
        workspace = cupy.empty(buffersize, dtype=dtype)

        # Cholesky Decomposition
        potrf(cusolver_handle, cublas.CUBLAS_FILL_MODE_UPPER, m,
              a.data.ptr, m, workspace.data.ptr, buffersize, dev_info.data.ptr)

        # TODO(y1r): check dev_info status

        # solve for the inverse
        potrs(cusolver_handle, cublas.CUBLAS_FILL_MODE_UPPER, m,
              m, a.data.ptr, m, b.data.ptr, m, dev_info.data.ptr)

        # TODO(y1r): check dev_info status

    return b
