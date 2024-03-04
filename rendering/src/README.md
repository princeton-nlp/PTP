# Renderer

Our renderer is written in c++ and can be compiled to a python package by using `pybind11`. We already compiled two versions for python 3.9 (`renderer.cpython-39-x86_64-linux-gnu.so`) and python 3.10 (`renderer.cpython-310-x86_64-linux-gnu.so`), which means that if you are using python 3.9/3.10, you do not need to do anything.

If you are using a different python version or find that our renderer does not work on your machine, you can compile it from scratch by simply running the `make` command in this folder. 