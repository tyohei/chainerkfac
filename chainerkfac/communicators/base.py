from chainermn.communicators import mpi_communicator_base
import warnings


class KfacCommunicatorBase(mpi_communicator_base.MpiCommunicatorBase):

    def __init__(self, mpi_comm):
        super(KfacCommunicatorBase, self).__init__(mpi_comm)
        self.indices = None

    def allreduce_grad(self):
        # We don't use AllReduce for training K-FAC
        warnings.warn('AllReduce called, skipping...')

    def reduce_scatter_v_arrays(self, arrays, stream=None):
        raise NotImplementedError

    def all_gather_v_arrays(self, arrays, stream=None):
        raise NotImplementedError
