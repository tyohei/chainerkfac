import warnings

from chainer.backends import cuda


def _check_array(array, name):
    xp = cuda.get_array_module(array)
    with cuda.get_device_from_array(array):
        if not array.dtype == xp.float32 and not array.dtype == xp.float16:
            warnings.warn('non FP32 or FP16 dtype detected in {}'.format(name))
            array = array.astype(xp.float32)
        if not (array.flags.c_contiguous or array.flags.f_contiguous):
            warnings.warn('non contiguous array detected in {}'.format(name))
            array = xp.ascontiguousarray(array)
    return array


def extract(fblocks, indices, extractors):
    """Extracts arrays from given fisher blocks using indices and extractors

    Args:
        fblocks: List of ``FisherBlock`` instances
        indices: List of ``int``s
        extractors: Callable that extracts arrays from a given ``FisherBlock``

    Return:
        List of tuple(array, bool). Second item indicates triangular flag.
    """
    arrays = []
    for local_indices in indices:
        local_arrays = []
        for index in local_indices:
            for extractor in extractors:
                for array in extractor(fblocks[index]):
                    local_arrays.append(array)
        arrays.append(local_arrays)
    return arrays


def extract_attr(attr, triangular=False):
    """Extracts arrays from a given ``FisherBlock``"""

    def _extract_attr(fblock):
        arrays = []
        target = getattr(fblock, attr, None)
        if target is not None:
            for i, x in enumerate(target):
                x = _check_array(x, fblock.linkname)
                target[i] = x
                arrays.append((x, triangular))
        return arrays

    return _extract_attr


def extract_attr_from_params(attr, triangular=False):
    """Extracts arrays from all ``Parameter``s in a given ``FisherBlock``"""

    def _extract_attr_from_params(fblock):
        arrays = []
        for _, param in sorted(fblock.link.namedparams()):
            x = getattr(param, attr, None)
            if x is not None:
                x = _check_array(x, fblock.linkname)
                setattr(param, attr, x)
                arrays.append((x, triangular))
        return arrays

    return _extract_attr_from_params


def get_nelems(arrays):
    """Computes number of elements from given arrays using the triangular flag."""  # NOQA
    nelems = 0
    for local_arrays in arrays:
        for array, triangular in local_arrays:
            if triangular:
                if array.shape[0] != array.shape[1]:
                    raise RuntimeError('get_nelems: not a square matrix')
                nelems += array.shape[0] * (array.shape[0] + 1) // 2
            else:
                nelems += array.size
    return nelems


def assign(gpu_buf, nbytes):
    if nbytes > gpu_buf.size:
        gpu_buf.assign(nbytes)
        return True
    return False


def allocate_asgrad(fblocks, attr):
    for fblock in fblocks:
        for _, param in sorted(fblock.link.namedparams()):
            if not hasattr(param, attr):
                # We need to allocate memory space for recieving data
                _grad = param.grad.copy()
                _grad.fill(0.)
                setattr(param, attr, _grad)
