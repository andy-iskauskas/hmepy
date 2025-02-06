===============
hmePy: History Matching and Emulation in Python
===============

Overview
========

The goal of hmepy is to make the process of history matching and emulation accessible and easily useable by modellers. The central object of the process is an *Emulator*: a statistical approximation for the output of a complex (and often expensive) model that, given a relatively small number of model evaluations, can give predictions of the model output at unseen points with appropriate uncertainty built in. Using these predictions, we may follow a process of *history matching*, where infeasible parts of the parameter space are ruled out. Sampling parameter combinations from this reduced space allows us to train more accurate emulators, which in turn remove more parameter space, and so on.

The *hmepy* package contains tools for the automated construction of emulators, validation diagnostics, and a careful point proposal mechanism to sample the remaining acceptable space.

Installation
============

Clone the `repo <https://github.com/andy-iskauskas/hmepy>`_ into an appropriate location. From terminal, navigate to the ``hmepy-main`` directory and run ``python3 -m pip install -e .``. This should install the package into the current environment.

Documentation
=============

Documentation for individual functions can be accessed via the usual methods: for example, ``help(emulatorFromData)``. A companion paper to the `R package <https://github.com/andy-iskauskas/hmer>`_ details the base functionality in more depth, including an example.
