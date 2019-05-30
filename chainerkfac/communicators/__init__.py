def create_communicator(communicator_name='pure_nccl', mpi_comm=None):
    if mpi_comm is None:
        import mpi4py.MPI
        mpi_comm = mpi4py.MPI.COMM_WORLD

    if communicator_name == 'pure_nccl':
        from chainerkfac.communicators.pure_nccl_communicator \
            import PureNcclCommunicator
        return PureNcclCommunicator(mpi_comm)
    else:
        raise ValueError(
            'Unrecognized communicator_name: {}'.format(communicator_name))
