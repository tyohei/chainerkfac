import math

import cupy
import numpy as np

import chainer

from chainermn.communicators import _communication_utility
from chainermn.communicators import _memory_utility
from chainermn import nccl

from chainerkfac.communicators import _utility
from chainerkfac.communicators import base


class PureNcclCommunicator(base.KfacCommunicatorBase):

    def __init__(self, mpi_comm):
        super(PureNcclCommunicator, self).__init__(mpi_comm)

        # We have to delay the initialization of communicators. This is because
        # NCCL's communicators use the current CUDA devices at the time of
        # initialization. Therefore, we have to initialize NCCL communicators
        # after users set the devices to use.
        self.nccl_comm = None

        # GPU buffers
        self.gpu_buf_a = _memory_utility.DeviceMemory()
        self.gpu_buf_b = _memory_utility.DeviceMemory()

        # Data type used in communications
        self._rsv_comm_dtype = np.dtype(np.float32)
        self._agv_comm_dtype = np.dtype(np.float32)

        # GPU kernels. We don't generate here due to the same reason above
        self._rsv_memset_kernel = None
        self._agv_memset_kernel = None

        # array info
        self._ainfo = None

    def _init_comms(self):
        if self.nccl_comm is not None:
            return
        self.nccl_comm = _communication_utility.init_nccl_comm(self.mpi_comm)

    def reduce_scatter_v_arrays(self, arrays, stream=None, debug=False):
        """Executes Reduce-Scatter-V.

            1. pack:   gbuf_A <- arrays
            2. send:    ....  <- gbuf_A
            3. recv:   gbuf_B <-  ....
            4. unpack: arrays <- gbuf_B

        """

        # CUDA default stream
        if stream is None:
            stream = chainer.cuda.Stream.null

        # Initialize NCCL communicator if not
        self._init_comms()

        # Target NCCL communicator
        nccl_comm = self.nccl_comm

        # This processes's assigned array index in arrays
        local_rank = self.rank

        # Get total number of elements, local number of elements, and local
        # number of elements' offset
        nelems = _utility.get_nelems(arrays)

        # Allocate memory if not
        needs_sync = False
        nbytes = nelems * self._rsv_comm_dtype.itemsize
        needs_sync |= _utility.assign(self.gpu_buf_a, nbytes)
        needs_sync |= _utility.assign(self.gpu_buf_b, nbytes)

        # Synchronize if necessary
        if needs_sync:
            stream.synchronize()
            chainer.cuda.Stream.null.synchronize()

        # Pack elements in a buffer
        # Data type casting will occur here if necessary
        target = -1  # All ranks
        self._pack_arrays_to_buffer(arrays, self.gpu_buf_a,
                                    self._rsv_comm_dtype, target,
                                    stream, debug)

        # Buffers for AllReduce
        sendbuf = self.gpu_buf_a.ptr()
        recvbuf = self.gpu_buf_b.ptr()

        # Communication
        nccl_dtype = _get_nccl_dtype(self._rsv_comm_dtype)
        nccl_comm.allReduce(sendbuf, recvbuf, nelems, nccl_dtype,
                            nccl.NCCL_SUM, stream.ptr)

        # Compute the mean (divide by the number of processes)
        # Unpack elements from a buffer
        # Data type casting will occur here if necessary
        divisor = self.size
        target = local_rank
        self._unpack_arrays_from_buffer(arrays, self.gpu_buf_b,
                                        self._rsv_comm_dtype, target,
                                        divisor, stream, debug)

    def _pack_arrays_to_buffer(self, arrays, gpu_buf, buf_dtype, target,
                               stream, debug=False):
        self._ainfo = _ArraysInfo(arrays)
        ainfo = self._ainfo
        if debug and self.rank == 0:
            ainfo.show()
        n_arrays = ainfo.n_arrays
        buf_dtype = _get_nccl_dtype(buf_dtype)
        total_threads = ainfo.size_total
        n_threads = 128
        n_blocks = (total_threads + n_threads - 1) // n_threads
        _cupy_batched_pack_arrays()(
            (n_blocks,), (n_threads,),
            (gpu_buf.memory.ptr, buf_dtype, ainfo.buf_size_csum,
             ainfo.dptr, ainfo.dtype, ainfo.size_csum,
             ainfo.triangle_size,
             ainfo.rank, target,
             total_threads, n_arrays), stream=stream)

    def _unpack_arrays_from_buffer(self, arrays, gpu_buf, buf_dtype, target,
                                   divisor, stream, debug=False):
        if self._ainfo is None:
            self._ainfo = _ArraysInfo(arrays)
        ainfo = self._ainfo
        if debug and self.rank == 0:
            ainfo.show()
        n_arrays = ainfo.n_arrays
        buf_dtype = _get_nccl_dtype(buf_dtype)
        total_threads = ainfo.size_total
        n_threads = 128
        n_blocks = (total_threads + n_threads - 1) // n_threads
        _cupy_batched_unpack_arrays()(
            (n_blocks,), (n_threads,),
            (ainfo.dptr, ainfo.dtype, ainfo.size_csum,
             gpu_buf.memory.ptr, buf_dtype, ainfo.buf_size_csum,
             ainfo.triangle_size,
             ainfo.rank, target, divisor,
             total_threads, n_arrays), stream=stream)
        self._ainfo = None

    def all_gather_v_arrays(self, arrays, stream=None, debug=False):
        """Executes All-Gather-V.

            0. memset: gbuf_A <- (zero)
            1. pack:   gbuf_A <- arrays
            2. send:    ....  <- gbuf_A
            3. recv:   gbuf_B <-  ....
            4. unpack: arrays <- gbuf_B

        """

        # CUDA default stream
        if stream is None:
            stream = chainer.cuda.Stream.null

        # This processes's assigned array index in arrays
        local_rank = self.rank

        # Initialize NCCL communicator if not
        self._init_comms()

        # Target NCCL communicator
        nccl_comm = self.nccl_comm

        # Get total number of elements, local number of elements, and local
        # number of elements' offset
        nelems = _get_divideable_nelems(nccl_comm, _utility.get_nelems(arrays))

        # Allocate memory if not
        needs_sync = False
        nbytes = nelems * self._agv_comm_dtype.itemsize
        needs_sync |= _utility.assign(self.gpu_buf_a, nbytes)
        needs_sync |= _utility.assign(self.gpu_buf_b, nbytes)

        # Synchronize if necessary
        if needs_sync:
            stream.synchronize()
            chainer.cuda.Stream.null.synchronize()

        # Memset 0
        if self._agv_memset_kernel is None:
            self._agv_memset_kernel = _get_memset_kernel(self._agv_comm_dtype)
        self._agv_memset_kernel(
            self.gpu_buf_a.array(nelems, dtype=self._agv_comm_dtype),
            stream=stream)

        # Pack elements in a buffer
        # Data type casting will occur here if necessary
        target = local_rank
        self._pack_arrays_to_buffer(arrays, self.gpu_buf_a,
                                    self._agv_comm_dtype, target, stream,
                                    debug)

        # Buffers for AllReduce
        sendbuf = self.gpu_buf_a.ptr()
        recvbuf = self.gpu_buf_b.ptr()

        # Communication
        nccl_dtype = _get_nccl_dtype(self._agv_comm_dtype)
        nccl_comm.allReduce(sendbuf, recvbuf, nelems, nccl_dtype,
                            nccl.NCCL_SUM, stream.ptr)

        # Unpack elements from a buffer
        # Data type casting will occur here if necessary
        divisor = 1
        target = -1  # all ranks
        self._unpack_arrays_from_buffer(arrays, self.gpu_buf_b,
                                        self._agv_comm_dtype, target, divisor,
                                        stream, debug)


class _ArraysInfo(object):
    def __init__(self, arrays):
        n_arrays = sum(len(local_arrays) for local_arrays in arrays)

        # array_id -> assigned_rank
        ainfo_rank = np.empty(n_arrays, dtype=np.int32)

        # array_id -> device_ptr
        ainfo_dptr = np.empty(n_arrays, dtype=np.int64)

        # array_id -> dtype (nccl enum value)
        ainfo_dtype = np.empty(n_arrays, dtype=np.int32)

        # array_id -> size of matrix if triangle_pack else 0
        ainfo_triangle_size = np.empty(n_arrays, dtype=np.int32)

        # cumulative sum
        ainfo_size_csum = np.empty(n_arrays + 1, dtype=np.int32)
        ainfo_buf_size_csum = np.empty(n_arrays + 1, dtype=np.int32)

        # initial value
        ainfo_size_csum[0] = 0
        ainfo_buf_size_csum[0] = 0

        i = 0
        for rank, local_arrays in enumerate(arrays):
            for array, triangular in local_arrays:
                ainfo_rank[i] = rank
                ainfo_dptr[i] = array.data.ptr

                if array.dtype not in [np.float16, np.float32]:
                    raise ValueError('dtype must be float16 or float32.')

                ainfo_dtype[i] = _get_nccl_dtype(array.dtype)

                if triangular:
                    _len = array.shape[0]
                    ainfo_triangle_size[i] = _len
                    nelems = _len * (_len + 1) // 2
                else:
                    ainfo_triangle_size[i] = 0
                    nelems = array.size

                ainfo_size_csum[i + 1] = ainfo_size_csum[i] + array.size
                ainfo_buf_size_csum[i + 1] = ainfo_buf_size_csum[i] + nelems
                i += 1

        assert (n_arrays == i)

        # Copy to device
        self.n_arrays = n_arrays
        self.size_total = ainfo_size_csum[n_arrays]
        self.buf_size_total = ainfo_buf_size_csum[n_arrays]
        self.rank = cupy.asarray(ainfo_rank)
        self.dptr = cupy.asarray(ainfo_dptr)
        self.dtype = cupy.asarray(ainfo_dtype)
        self.triangle_size = cupy.asarray(ainfo_triangle_size)
        self.size_csum = cupy.asarray(ainfo_size_csum)
        self.buf_size_csum = cupy.asarray(ainfo_buf_size_csum)

    def show(self):
        print('# ainfo: n_arrays: {}'.format(self.n_arrays))
        print('# ainfo: size_total: {}'.format(self.size_total))
        print('# ainfo: buf_size_total: {}'.format(self.buf_size_total))
        for i in range(self.n_arrays):
            size = self.size_csum[i + 1] - self.size_csum[i]
            buf_size = self.buf_size_csum[i + 1] - self.buf_size_csum[i]
            triangle_size = self.triangle_size[i]
            print('# ainfo[{}], rank:{}, dptr:{}, dtype:{}, size:{}, '
                  'buf_size:{}, triangle_size:{}'
                  ''.format(i, self.rank[i], self.dptr[i], self.dtype[i],
                            size, buf_size, triangle_size))
            if triangle_size > 0:
                assert (size == triangle_size ** 2)
                assert (buf_size == triangle_size * (triangle_size + 1) // 2)
            else:
                assert (size == buf_size)


def _cupy_batched_pack_arrays():
    return cupy.RawKernel(r'''
#include <cuda_fp16.h>
#define NCCL_FLOAT16  6
#define NCCL_FLOAT32  7
    extern "C" __global__
    void cupy_batched_pack_arrays(
        void *dst_head,
        int dst_dtype,
        const int *dst_size_csum,
        const unsigned long *list_src_dptr,
        const int *list_src_dtype,
        const int *src_size_csum,
        const int *triangle_size,
        const int *list_rank, int rank,
        int total_threads, int n_arrays)
    {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid >= total_threads) return;
        int j_min = 0;
        int j_max = n_arrays - 1;
        int j;
        while (1) {
            j = (j_min + j_max) / 2;
            if (tid < src_size_csum[j]) {
                j_max = j - 1;
                continue;
            }
            if (tid >= src_size_csum[j+1]){
                j_min = j + 1;
                continue;
            }
            break;
        }
        assert(tid >= src_size_csum[j]);
        assert(tid < src_size_csum[j+1]);
        if (rank >= 0 && list_rank[j] != rank) return;

        unsigned long src_dptr = list_src_dptr[j];
        int src_dtype = list_src_dtype[j];
        int src_idx = tid - src_size_csum[j];
        int dst_idx;
        int matrix_size = triangle_size[j];
        if (matrix_size > 0) {
            // triangle pack
            int row_idx = src_idx / matrix_size;
            int col_idx = src_idx % matrix_size;
            if (col_idx > row_idx) return;
            dst_idx = (row_idx * (row_idx+1) / 2) + col_idx;
        }
        else {
            dst_idx = src_idx;
        }

        if (dst_dtype == NCCL_FLOAT16) {
            half* dst = (half*)dst_head + dst_size_csum[j];
            if (src_dtype == NCCL_FLOAT16) {
                half* src = (half*)src_dptr;
                dst[dst_idx] = (half) src[src_idx];
            }
            else if (src_dtype == NCCL_FLOAT32) {
                float* src = (float*)src_dptr;
                dst[dst_idx] = (half) src[src_idx];
            }
        }
        else if (dst_dtype == NCCL_FLOAT32) {
            float* dst = (float*)dst_head + dst_size_csum[j];
            if (src_dtype == NCCL_FLOAT16) {
                half* src = (half*)src_dptr;
                dst[dst_idx] = (float) src[src_idx];
            }
            else if (src_dtype == NCCL_FLOAT32) {
                float* src = (float*)src_dptr;
                dst[dst_idx] = (float) src[src_idx];
            }
        }
    }
    ''', 'cupy_batched_pack_arrays')


def _cupy_batched_unpack_arrays():
    return cupy.RawKernel(r'''
#include <cuda_fp16.h>
#define NCCL_FLOAT16  6
#define NCCL_FLOAT32  7
    extern "C" __global__
    void cupy_batched_unpack_arrays(
        unsigned long *list_dst_dptr,
        const int *list_dst_dtype,
        const int *dst_size_csum,
        const void *src_head,
        int src_dtype,
        const int *src_size_csum,
        const int *triangle_size,
        const int *list_rank, int rank, int divisor,
        int total_threads, int n_arrays)
    {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid >= total_threads) return;
        int j_min = 0;
        int j_max = n_arrays - 1;
        int j;
        while (1) {
            j = (j_min + j_max) / 2;
            if (tid < dst_size_csum[j]) {
                j_max = j - 1;
                continue;
            }
            if (tid >= dst_size_csum[j+1]){
                j_min = j + 1;
                continue;
            }
            break;
        }
        assert(tid >= dst_size_csum[j]);
        assert(tid < dst_size_csum[j+1]);
        if (rank >= 0 && list_rank[j] != rank) return;

        unsigned long dst_dptr = list_dst_dptr[j];
        int dst_dtype = list_dst_dtype[j];
        int dst_idx = tid - dst_size_csum[j];
        int src_idx;
        int matrix_size = triangle_size[j];
        if (matrix_size > 0) {
            // triangle pack
            int row_idx = dst_idx / matrix_size;
            int col_idx = dst_idx % matrix_size;
            if (row_idx >= col_idx) {
                src_idx = (row_idx * (row_idx+1) / 2) + col_idx;
            }
            else {
                src_idx = (col_idx * (col_idx+1) / 2) + row_idx;
            }
        }
        else {
            src_idx = dst_idx;
        }

        if (src_dtype == NCCL_FLOAT16) {
            half* src = (half*)src_head + src_size_csum[j];
            half src_val = src[src_idx] / (half) divisor;
            if (dst_dtype == NCCL_FLOAT16) {
                half* dst = (half*)dst_dptr;
                dst[dst_idx] = (half) src_val;
            }
            else if (dst_dtype == NCCL_FLOAT32) {
                float* dst = (float*)dst_dptr;
                dst[dst_idx] = (float) src_val;
            }
        }
        else if (src_dtype == NCCL_FLOAT32) {
            float* src = (float*)src_head + src_size_csum[j];
            float src_val = src[src_idx] / (float) divisor;
            if (dst_dtype == NCCL_FLOAT16) {
                half* dst = (half*)dst_dptr;
                dst[dst_idx] = (half) src_val;
            }
            else if (dst_dtype == NCCL_FLOAT32) {
                float* dst = (float*)dst_dptr;
                dst[dst_idx] = (float) src_val;
            }
        }
    }
    ''', 'cupy_batched_unpack_arrays')


def _get_memset_kernel(dtype):
    return chainer.cuda.cupy.ElementwiseKernel(
        '',
        '{} x'.format(dtype.name),
        'x = 0.0',
        'my_memset')


def _get_divideable_nelems(nccl_comm, nelems):
    if hasattr(nccl_comm, 'getCountRequirement'):
        requirement = nccl_comm.getCountRequirement()
        return int(math.ceil(nelems / requirement)) * requirement
    else:
        return nelems


def _get_nccl_dtype(dtype):
    if dtype == np.float16:
        return nccl.NCCL_FLOAT16
    elif dtype == np.float32:
        return nccl.NCCL_FLOAT32
    elif dtype == np.float64:
        return nccl.NCCL_FLOAT64
    elif dtype == np.uint64:
        return 9  # PackedFloat21
    else:
        raise ValueError(
            'dtype must be numpy.float16, numpy.float32 or numpy.float64,'
            'not {}'.format(dtype))
