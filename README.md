# Convolutions

## Introduction

A collection of convolution algorithms accelerated with
[NumPy](https://numpy.org/), [CuPy](https://cupy.dev/) and
[Numba](https://numba.pydata.org/). Written for instructive purposes and to
complement the somewhat lacking documentation on how to write [CUDA kernels](
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#kernels) in
CuPy and Numba, and it does so with a motivating/fun application -- and no,
it's not Convolutional NNs 😒

## Description

Convolution is used here to evolve a 
[Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) 
(the prototypical Cellular Automaton model). CAs, which compute the viability 
of a cell (ie a binary value in a 2D grid) based on the values of the 8 
neighbouring cells (0 = dead, 1 = alive) , naturally benefit from the 
optimizations used in NumPy's various convolution algorithms.

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


### Notes

-   If you like CAs (who doesn't), check out 
    [my other project](https://github.com/GregSotiropoulos/cellular_automata),
    which has a full-fledged GUI that allows you to run, visualize and save
    CAs, among other things.
-   To use CUDA functionality (CuPy, Numba's ``cuda.jit``), you need to
    install the appropriate version of the CUDA Toolkit and CuPy. Head
    over to CuPy's 
    [installation guide](https://docs.cupy.dev/en/stable/install.html) 
    for details. The existing functions have been tested on CUDA 11.0 and 11.1.
-   Since there is no ``requirements.txt`` or ``setup.py``, users should be
    aware that on Windows x64 system, NumPy must be of version other
    than 1.19.4 -- for the (gory) details of why this is, see 
    [here](https://github.com/numpy/numpy/wiki/FMod-Bug-on-Windows).