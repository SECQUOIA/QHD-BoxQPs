This repository contains benchmark box-constrained quadratic programs.

Problem is given by:  
    min x' M x  
    s.t. 0 <= x <= 1  

M is saved as a sparse matrix in .npz format.  

# verification
NOTE: solutions for this dataset were found using Anstreicher's MILP formulation for checking copositivity (https://optimization-online.org/2020/03/7659/). This may not correspond to the global optimum of the BoxQP formulation above.

summary.csv contains a list of summary information about the instances:  
    nVars (side length of M),  
    nLayer (number of hidden layers),  
    nHidden (number of hidden nodes per layer),  
    idx (iteration in cutting plane algorithm this instance is from),  
    objVal (best found objective found using Anstreicher's MILP formulation),  
    runtime(s) (runtime of the copositivity checks in seconds) . 

The corresponding M file is given by 'M_nlayers={}_nhidden={}_idx={}.npz'.format(nlayers, nhidden, idx).  
The solution corresponding to objVal is given by 'sol_nlayers={}_nhidden={}_idx={}.npy'.format(nlayers, nhidden, idx). 

# max_clique
NOTE: solutions for this dataset were found using Anstreicher's MILP formulation for checking copositivity (https://optimization-online.org/2020/   03/7659/). This may not correspond to the global optimum of the BoxQP formulation above. Most solutions end up being naturally binary.

summary.csv contains a list of summary information about the instances:  
    n (side length of M, ranges from 10, 30, 50, ... 130),  
    p (density of the graph: 0.25, 0.5, 0.75),  
    N (instance number: 1, ... 25),  
    idx (iteration in cutting plane algorithm this instance is from),  
    objVal (best found objective found using Anstreicher's MILP formulation),  
    runtime(s) (runtime of the copositivity checks in seconds) . 

The corresponding M file is given by 'M_n={}_p={}_N={}_idx={}.npz'.format(n, p, N, idx).  
The solution corresponding to objVal is given by 'sol_n={}_p={}_N={}_idx={}.npy'.format(n, p, N, idx).  
