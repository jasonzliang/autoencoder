Stacked Denoising Autoencoders
by Jason Liang and Keith Kelly

Features:
-C++ implementation of feed-forward neural networks and stacked denoising autoencoders
-MultiThreaded and Fast: Order of magnitude faster than training with Theano


To build with OpenMP:
1)cd to src directory and type "make"

To build with OpenBlas (you must have it installed):
1)cd to src directory
2)in the Makefile, set OPEN_BLAS_INC and OPEN_BLAS_LIB appropriately
3)type "make HAS_OPENBLAS=1"

Project Report:
http://users.ices.utexas.edu/~keith/files/autoencoder/final_report/autoencoder.pdf

Datasets can be found at: 
users.ices.utexas.edu/~keith/files/MLProject.html


