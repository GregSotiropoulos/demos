# coding=utf-8

"""
A collection of convolution algorithms accelerated with
`NumPy <https://numpy.org/>`_, `CuPy <https://cupy.dev/>`_ and
`Numba <https://numba.pydata.org/>`_. Written for instructive purposes and to
complement the somewhat lacking documentation on how to write `CUDA kernels
<https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#kernels>`_
in CuPy and Numba, and it does so with a motivating/fun application -- and no,
it's not Convolutional NNs 😒

|
Convolution is used here to evolve a Game of Life (the prototypical Cellular
Automaton model). CAs, which compute the viability of a cell (ie a binary value
in a 2D grid) based on the values (0 = dead, 1 = alive) of the 8 neighbouring
cells, naturally benefit from the optimizations used in NumPy's various
convolution algorithms.

|
There are multiple convolution methods in NumPy/SciPy and each of them can be
further accelerated with CuPy or Numba. So how are we to choose one? This
module aims to provide insights on this question in an easy and (hopefully)
fun way. Apart from sheer speed, relevant issues that will emerge through
experimentation are:
    *   Compatibility/universality: not all methods work on all systems. For
        starters, to use the GPU-accelerated versions, you need a working CUDA
        Developer's Toolkit installation. Furthermore, this code is untested on
        Linux-based systems (although it is expected to work).
    *   (Not so) subtle issues with array sizes/alignment -- for example there
        seems to be an issue with CuPy's ``RawKernel`` method for arrays whose
        dimensions are not powers of 2. I'm currently investigating this and
        will have something to say about it (a fix and/or explanation) in the
        update.
    *   Choice of certain parameters, such as the "threads per block" in CUDA
        kernels.

|
Notes
=====
    *   If you like CAs (who doesn't), check out `my other project
        <https://github.com/GregSotiropoulos/cellular_automata>`_,
        which has a full-fledged GUI that allows you to run, visualize and save
        CAs, among other things.
    *   To use CUDA functionality (CuPy, Numba's ``cuda.jit``), you need to
        install the appropriate version of the CUDA Toolkit and CuPy. Head
        over to `CuPy's installation guide
        <https://docs.cupy.dev/en/stable/install.html>`_ for details. The
        existing functions have been tested on CUDA 11.0 and 11.1.
    *   Since there is no ``requirements.txt`` or ``setup.py``, users should be
        aware that on Windows x64 system, NumPy has to have a version other
        than 1.19.4 -- for the (gory) details of why this is, see `here
        <https://github.com/numpy/numpy/wiki/FMod-Bug-on-Windows>`_.
"""

# NumPy
import numpy as np
from scipy.ndimage import convolve as conv_np_ndi
from scipy.signal import convolve2d as conv_np_sig_2d
from scipy.ndimage import uniform_filter as uf_np

# CuPy
import cupy as cp
from cupyx.scipy.ndimage import uniform_filter as uf_cp, convolve as conv_cp_ndi
from cupyx.scipy.signal import convolve2d as conv_cp_sig_2d

# Numba
import numba as nb
from numba import cuda, stencil

# stdlib
import sys
from time import perf_counter as t
from functools import partial
from itertools import chain
from math import ceil
from operator import attrgetter, sub, rshift
import re
from types import SimpleNamespace as Sns
from collections.abc import Mapping


def default_options():
    """
    Get default configuration options (see ``options()``). The *_grades options
    refer to the grid dimensions (shape_grades) and the number of generations
    that the CA (cellular automaton) grid is to evolve for. 'Fast', 'slow' and
    'mixed' refer to the parameter values used when the functions in the set to
    be compared (determined by the wildcard/regex pattern of function names,
    as shown in ``main()``) are all GPU-accelerated (so CuPy or Numba with
    CUDA support),

    :return: A dictionary of default options.
    """
    return dict(
        # CA grid shape
        shape_grades=dict(
                fast=(1024*4,)*2,
                mixed=(1024*2,)*2,
                slow=(1024,)*2
        ),
        # number of generations to compute
        gens_grades=dict(
                fast=50,
                mixed=100,
                slow=100
        ),
        int_types=('u1', 'i1'),
        threads_per_block=(8, ) * 2,  # CUDA's blockDim
        random_seed=0,
        run_func_prefix='run_ca_'
    )


# this is the global (module-level) variable that holds the options
# (configuration) -- see options() function below
conv_opts = Sns()


def all_func_speeds(fnames=(), **opts):
    """
    Determine whether the functions (passed as function names in ``fnames``)
    to be tested are all fast, all slow or a mix of fast and slow ones.

    :param fnames: A tuple of function names.
    :param opts: Additional options.
    :return: A dictionary of options. If `opts` was non-empty, the dictionary
        includes those options, except for the 'fnames' and 'grade' items.
    """
    opts['fnames'] = fns = *(
        fn.strip().replace('run_ca_', '')
        for fn in chain(
            fnames,
            opts.get('fnames', ()),
            map(attrgetter('__name__'), opts.get('funcs', ()))
        )
    ),
    fast = *map(partial(re.search, r'(?:^\s*(cp_\w*|\w*cuda))(.*)\s*$'), fns),
    if fast and all(fast):
        opts['grade'] = 'fast'
        return opts
    slow = map(partial(re.search, r'^\s*(np_manual|\w*sten)\s*$'), fns)
    if any(slow):
        opts['grade'] = 'slow'
    else:
        opts['grade'] = 'mixed'
    return opts


def options(*args, **kw):
    """
    Get various options (configuration) as a nested namespace, so that
    something like ``opts.a.b.c`` could be used. Note that the `data` top-level
    attribute contains the initial inputs (starting grids) of the
    :param args: Mappings (dictionaries) that may contain options. Non-mapping
        arguments are silently discarded.
    :param kw: Additional keywords. They override any items of the same name
        present in any of the *args dictionaries.
    :return: A nested namespace of options.
    """

    # Attribute names for sub-namespaces.
    np_cp_s = 'np', 'cp'  # NumPy, CuPy
    tps_s = 'u', 's'  # unsigned, signed

    o = Sns(**default_options())  # initialize namespace

    od = vars(o)
    np_cp = dict(np=np, cp=cp)
    for a in (*args, kw):
        if isinstance(a, Mapping):
            od.update(a)

    od.update(all_func_speeds(**od))
    for sg in ('shape', 'gens'):
        if sg not in od:
            od[sg] = od[f'{sg}_grades'][o.grade]
            print('setting', sg, 'to', getattr(o, sg))
    o.h, o.w = o.shape

    o.data = Sns(np=Sns(), cp=Sns())
    odd = vars(o.data)
    o.blocks_per_grid = *(
        ceil(tpb / hw) for (tpb, hw) in zip(o.shape, o.threads_per_block)
    ),
    npr = np.random
    npr.seed(o.random_seed)
    init_np = npr.randint(0, 2, o.shape, dtype=o.int_types[0])
    for xp_s in np_cp_s:  # for each ('np', 'cp')
        xp = np_cp[xp_s]  # actual NumPy or CuPy module
        odxp = vars(odd[xp_s])
        for i, tp_s in enumerate(tps_s):  # for each 'u' (unsigned) or 's'
            odxp[tp_s] = data = Sns()
            data.dtype = dtp = xp.dtype(o.int_types[i])
            data.init = xp.array(init_np).astype(dtp)
            data.rng = xp.arange(5, 8).astype(dtp)
            data.isin = xp.isin
            data.ker = ker = xp.full((3, 3), 2-i, dtype=dtp)
            ker[1, 1] = 1
            data.ker_flat = ker.flatten()

            data.grid = xp.mgrid[:o.h, :o.w]
            #
            data.ker_idxs = map(
                sub, xp.indices(ker.shape), map(rshift, ker.shape, (1, 1))
            )
            data.idxs = *(
                (
                    idx +
                    xp.broadcast_to(kidx.flatten(), (o.h, o.w, ker.size)).T
                ) % d
                for (idx, kidx, d) in zip(data.grid, data.ker_idxs, o.shape)
            ),

    global conv_opts
    if 'conv_opts' not in globals() or not isinstance(conv_opts, Sns):
        conv_opts = o
    vars(conv_opts).update(od)

    return conv_opts


def run_ca_np_manual():
    o = conv_opts.data.np.s
    ker, out, idxs = o.ker_flat, o.init.copy(), o.idxs
    f = lambda: (out[idxs].T @ ker).T
    for i in range(conv_opts.gens):
        out[:, :] = (f() - 3) >> out == 0
    return out


def run_ca_np_uf():
    out = conv_opts.data.np.s.init.copy()
    f = lambda: uf_np(out*9, mode='wrap')
    for i in range(conv_opts.gens):
        out[:, :] = (f() - 3) >> out == 0
    return out


def run_ca_np_ndi_1():
    od = conv_opts.data.np.u
    isin, rng, ker, out = od.isin, od.rng, od.ker, od.init.copy()
    f = lambda: conv_np_ndi(out, ker, mode='wrap')
    for i in range(conv_opts.gens):
        out[:, :] = isin(f(), rng)
    return out


def run_ca_np_ndi_2():
    od = conv_opts.data.np.s
    ker, out = od.ker, od.init.copy()
    f = lambda: conv_np_ndi(out, ker, mode='wrap')
    for i in range(conv_opts.gens):
        out[:, :] = (f() - 3) >> out == 0
    return out.astype(conv_opts.data.np.u.dtype)


def run_ca_np_sig_2d_1():
    od = conv_opts.data.np.u
    isin, rng, ker, out = od.isin, od.rng, od.ker, od.init.copy()
    f = lambda: conv_np_sig_2d(out, ker, boundary='wrap', mode='same')
    for i in range(conv_opts.gens):
        out[:, :] = isin(f(), rng)
    return out


def run_ca_np_sig_2d_2():
    od = conv_opts.data.np.s
    ker, out = od.ker, od.init.copy()
    f = lambda: conv_np_sig_2d(out, ker, boundary='wrap', mode='same')
    for i in range(conv_opts.gens):
        out[:, :] = (f() - 3) >> out == 0
    return out.astype(conv_opts.data.np.u.dtype)


def run_ca_cp_sig_2d_1():
    od = conv_opts.data.cp.u
    isin, rng, ker, out = od.isin, od.rng, od.ker, od.init.copy()
    f = lambda: conv_cp_sig_2d(out, ker, boundary='wrap', mode='same')
    for i in range(conv_opts.gens):
        out[:, :] = isin(f(), rng)
    return cp.asnumpy(out)


def run_ca_cp_sig_2d_2():
    od = conv_opts.data.cp.s
    ker, out = od.ker, od.init.copy()
    f = lambda: conv_cp_sig_2d(out, ker, boundary='wrap', mode='same')
    for i in range(conv_opts.gens):
        out[:, :] = (f() - 3) >> out == 0
    return cp.asnumpy(out)


def run_ca_cp_ndi_1():
    od = conv_opts.data.cp.u
    isin, rng, ker, out = od.isin, od.rng, od.ker, od.init.copy()
    f = lambda: conv_cp_ndi(out, ker, mode='wrap')
    for i in range(conv_opts.gens):
        out[:, :] = isin(f(), rng)
    return cp.asnumpy(out)


def run_ca_cp_ndi_2():
    od = conv_opts.data.cp.s
    ker, out = od.ker, od.init.copy()
    f = lambda: conv_cp_ndi(out, ker, mode='wrap')
    for i in range(conv_opts.gens):
        out[:, :] = (f() - 3) >> out == 0
    return cp.asnumpy(out)


def run_ca_cp_uf():
    out = conv_opts.data.cp.u.init.copy()
    f = lambda: uf_cp(out*9, mode='wrap')
    for i in range(conv_opts.gens):
        out[:, :] = (f() - 3) >> out == 0
    return cp.asnumpy(out)


def run_ca_cp_raw_ker():
    o = conv_opts
    od = o.data.cp.u
    both = od.init.copy(), od.init.copy()
    h, w = o.shape

    cp_ker = cp.RawKernel(
        # kernel header
        fr'''
        #define uchar unsigned char
        #define N size_t({h})
        #define M size_t({w})
        extern "C" __global__ void cp_ker_ca(uchar in[N][M], uchar out[N][M])
        '''
        # kernel body
        r'''
        {     
            //int h = blockDim.x * gridDim.x;
            //int w = blockDim.y * gridDim.y;
            int i = blockIdx.x * blockDim.x + threadIdx.x;        
            int j = blockIdx.y * blockDim.y + threadIdx.y;
            if (i < N && j < M) 
            {
                int a, b;
                char v = -3;        
                for (a = i-1; a < i+2; ++a)
                    for (b = j-1; b < j+2; ++b)
                        v += in[a % N][b % M];
                out[i][j] = v >> in[i][j] == 0;
            }
        }
        ''',
        'cp_ker_ca'
    )
    j = 1
    args = o.blocks_per_grid, o.threads_per_block
    for _ in range(o.gens):
        cp_ker(*args, both[::j])
        j = -j
    return cp.asnumpy(both[::j][0])

# @cp.fuse(kernel_name='cp_ker')
# def cp_ker(ii, jj):
#    pass


def run_ca_nb_cuda_jit_1():
    """

    :return:
    """
    o = conv_opts
    od = o.data.cp.u
    both = od.init.copy()
    both = both, both.copy()
    h, w = od.init.shape

    @cuda.jit
    def conv_numba_cuda_1(in_, out_):
        """
        Of all the different kernel variants that I tried (and I tried a LOT)
        this one, with this particular indexing, seems to be the fastest:
        ranging from ii (or jj) to ii+3 is measurably faster than the range
        from ii-1 to to ii+2, even though the former requires the setting of
        the ij variables (in the latter we can just use ii, jj for the final
        indexing where the bitshift is.

        Note that the situation is reversed in
        the case of the raw kernel -- there the ii-1-based indexing is
        slightly faster!

        As a general rule, writing to memory (even setting simple scalar
        variables) is noticeable faster than skipping precomputing and instead
        redoing modulo etc operations (as is evident in the code below). As
        long as it's not some ridiculously complex operation, repeating them
        is usually faster than storing intermediate results. This seems to be
        especially the case in C/C++ code; Python, on the othe hand, often
        benefits from storing intermediate variables (presumably because it
        reduces stack usage?)

        :param in_:
        :param out_:
        """
        ii, jj = ij = cuda.grid(2)
        if ii < h and jj < w:
            v = -3
            for i in range(ii-1, ii+2):
                for j in range(jj-1, jj+2):
                    v += in_[i % h, j % w]
            out_[ij] = v >> in_[ij] == 0

    f = conv_numba_cuda_1[o.blocks_per_grid, o.threads_per_block]
    j = 1
    for _ in range(o.gens):
        f(*both[::j])
        j = -j
    return cp.asnumpy(both[::j][0])


def run_ca_nb_cuda_jit_2():
    """

    :return: NumPy (not CuPy) array with the last grid of the CA evolution
    """
    o = conv_opts
    od = o.data.cp.u
    both = od.init.copy()
    both = both, both.copy()
    h, w = od.init.shape
    #idx = *(np.arange(-1, x+1).astype(cp.int32) % x for x in (h, w)),

    idxs = cp.arange(-1, h+1) % h, cp.arange(-1, w+1) % w
    # cool way to construct an mgrid-like thing with a single complex 2darray
    # ca = np.mgrid[-1:2, -1:2].T @ [1, 1.j]

    @cuda.jit
    def conv_numba_cuda_2(in_, out_, idx_i, idx_j):
        """
        Of all the different kernel variants that I tried (and I tried a LOT)
        this one, with this particular indexing, seems to be the fastest:
        ranging from ii (or jj) to ii+3 is measurably faster than the range
        from ii-1 to to ii+2, even though the former requires the setting of
        the ij variables (in the latter we can just use ii, jj for the final
        indexing where the bitshift is.

        Note that the situation is reversed in
        the case of the raw kernel -- there the ii-1-based indexing is
        slightly faster!

        As a general rule, writing to memory (even setting simple scalar
        variables) is noticeable faster than skipping precomputing and instead
        redoing modulo etc operations (as is evident in the code below). As
        long as it's not some ridiculously complex operation, repeating them
        is usually faster than storing intermediate results. This seems to be
        especially the case in C/C++ code; Python, on the othe hand, often
        benefits from storing intermediate variables (presumably because it
        reduces stack usage?)

        :param in_:
        :param out_:
        :param idx_i:
        :param idx_j:
        """
        ii, jj = ij = cuda.grid(2)
        if ii < h and jj < w:
            v = -3
            for i in range(ii, ii+3):
                for j in range(jj, jj+3):
                    v += in_[idx_i[i], idx_j[j]]
            out_[ij] = v >> in_[ij] == 0

    j = 1
    f = lambda: conv_numba_cuda_2[o.blocks_per_grid, o.threads_per_block](
        *both[::j], *idxs
    )
    for _ in range(o.gens):
        f()
        j = -j
    return cp.asnumpy(both[::j][0])


def run_ca_nb_njit_sten():
    """
    TODO: document this HARD -- very counterintuitive!

    :return: NumPy (not CuPy) array with the last grid of the CA evolution
    """
    o = conv_opts
    data = o.data.np.u
    out1 = data.init.copy()

    @stencil(neighborhood=((0, 2), ) * 2)
    def conv_numba_sten(in1):
        """

        :param in1:
        :return:
        """
        v, r = -3, range(3)
        for i in r:
            for j in r:
                v += in1[i, j]
        return v >> in1[1, 1] == 0

    for _ in range(o.gens):
        # np.pad(out1, 1, mode='wrap') is equivalent to:
        # out2 = np.pad(out1, 1)  # pad regularly (with zeros)
        # out2[1:-1, 1:-1] = out1
        # out2[1:-1, 0] = out1[:, -1]
        # out2[1:-1, -1] = out1[:, 0]
        # out2[0, 1:-1] = out1[-1, :]
        # out2[-1, 1:-1] = out1[0, :]
        # out2[0, 0] = out1[-1, -1]
        # out2[-1, -1] = out1[0, 0]
        # out2[0, -1] = out1[-1, 0]
        # out2[-1, 0] = out1[0, -1]
        out2 = np.pad(out1, 1, mode='wrap')
        conv_numba_sten(out2, out=out1)
    return out1


# print results
def main(pat='*', print_arrays=False, **opts):
    """
    Called when module is run as a script.

    :param pat: Wildcard or regular expression pattern. Only functions whose
        names (stripped of the prefix 'run_ca_') match the pattern are included
        in the benchmark. If a pattern contains only alphanumeric/underscore
        and wildcard ('*' and '?') characters only, it is considered a wildcard
        pattern (which is automatically converted to a regular expression).
    :param print_arrays:
    :param opts:
    :return:
    """

    # if the pattern consists only of alphanumeric characters and/or
    # underscores and/or '*' (wildcard matching zero or more characters)
    # and/or '?' (wildcard matching exactly one character), it is considered
    # a wildcard pattern (which is then converted to a regex pattern) otherwise
    # a regex.
    if re.match(r'^[\w*?]*$', pat):
        if not pat.startswith('*'):
            pat = '^' + pat
        if not pat.endswith('*'):
            pat += '$'
        pat = pat.replace('*', '.*').replace('?', '.')

    print(
        'Function name regex',
        '-------------------',
        pat,
        end='\n\n', sep='\n'
    )

    o = options(opts)
    fn_prefix = o.run_func_prefix
    o.fnames = fnames = *filter(
        partial(re.search, pat),
        (
            fn[len(fn_prefix):]
            for fn in vars(sys.modules[__name__]) if fn.startswith(fn_prefix)
        )
    ),

    print(*fnames, '\nGrade: ' + o.grade, sep='\n', end='\n\n')

    gens = o.gens
    n_fn = len(fnames)
    arg = f'({gens})'
    space = len(max(fnames, key=len)) + len(arg)

    # grids, call strings (for display purposes), execution times
    arrays, call_str, dts = [], [], []
    for fn in fnames:
        f = eval(fn_prefix + fn)
        t0, arr = t(), f()
        dt = t() - t0
        dts.append(dt)
        arrays.append(cp.asnumpy(arr))
        call_str.append(f'{fn}{arg}'.center(space))
        print(f'{f.__name__}:'.ljust(space+11), f'{dt:.3f} sec')

    print()

    for i in range(n_fn):
        j = (i+1) % n_fn
        # t0 = t()
        print(
            call_str[i],
            f'{"!="[int(np.array_equal(arrays[i], arrays[j]))]}=',
            call_str[j]
        )
    print()
    # print(f'Compare-arrays dt: {t()-t0:.3f}')

    if print_arrays:
        print(conv_opts.data.np.u.init, '\n=================')
        for a, c in zip(arrays, call_str):
            print(f'{c}\n{a}\n=================')

    return arrays, call_str


if __name__ == '__main__':
    arrays, call_str = main(
        r'cuda|^cp_',
        #r'manual|np_ndi',
        #r'nb*',
        #r'cp.*(ndi|raw)|np_ndi|np_uf',
        #r'np_ndi*',
        print_arrays=False,
        gens=1,
        shape=(1024*2,)*2,
    )
