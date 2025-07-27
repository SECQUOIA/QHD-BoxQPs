import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Tuple, Optional
from qhdopt import QHD

class BoxQPSolver:
    """
    A class to solve Box-constrained Quadratic Programs using QHDOPT.
    The Box-constrained Quadratic Program problem format is:
        min x^T M x
        s.t. 0 <= x <= 1
    
    Where M is loaded from .npz files (sparse matrices)
    """
    
    def __init__(self, data_dir: str = "BoxQP_benchmarks-main/max_clique"):
        self.data_dir = data_dir

    
    def load_instance(self, n: int, p: float, N: int, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Load a specific QP instance from .npz file and its reference solution from .npy file.
        
        Args:
            n: Problem dimension (number of variables)
            p: Graph density parameter (0.25, 0.5, 0.75)
            N: Instance number
            idx: Index from cutting plane algorithm
            
        Returns:
            Tuple of (M_matrix, reference_solution)
        """
        # Load the matrix M from .npz file
        matrix_filename = f"M_n={n}_p={p}_N={N}_idx={idx}.npz"
        matrix_path = os.path.join(self.data_dir, matrix_filename)
        
        if not os.path.exists(matrix_path):
            raise FileNotFoundError(f"Matrix file not found: {matrix_path}")
        
        M_sparse = sp.load_npz(matrix_path)
        M = M_sparse.toarray()  # Convert to dense for QHDOPT
        
        # Load the reference solution from .npy file
        sol_filename = f"sol_n={n}_p={p}_N={N}_idx={idx}.npy"
        sol_path = os.path.join(self.data_dir, sol_filename)
        
        reference_solution = None
        if os.path.exists(sol_path):
            reference_solution = np.load(sol_path)
        
        return M, reference_solution

    def round_to_binary(self, x: np.ndarray, M: np.ndarray, method: str = "nearest") -> np.ndarray:
        """
        Round continuous solution to binary.
        
        Args:
            x: Continuous solution from QHDOPT
            M: Quadratic matrix
            method: Rounding method ("nearest", "threshold", "local_search", "iterative")
            
        Returns:
            Binary solution
        """
        if method == "nearest":
            return np.round(x).astype(int)
        
        elif method == "threshold":
            return (x >= 0.5).astype(int)
        
        elif method == "local_search":
            x_binary = np.round(x).astype(int)
            current_obj = x_binary.T @ M @ x_binary
            improved = True
            
            while improved:
                improved = False
                for i in range(len(x_binary)):
                    x_temp = x_binary.copy()
                    x_temp[i] = 1 - x_temp[i]  # Flip bit
                    new_obj = x_temp.T @ M @ x_temp
                    
                    if new_obj < current_obj:
                        x_binary = x_temp
                        current_obj = new_obj
                        improved = True
                        break
            
            return x_binary
        
        elif method == "iterative":
            methods = ["nearest", "threshold"]
            best_solution = None
            best_obj = float('inf')
            
            for sub_method in methods:
                candidate = self.round_to_binary(x, M, method=sub_method)
                obj = candidate.T @ M @ candidate
                
                if obj < best_obj:
                    best_obj = obj
                    best_solution = candidate
            
            if best_solution is not None:
                return self.round_to_binary(best_solution.astype(float), M, method="local_search")
            else:
                return np.round(x).astype(int)
        
        else:
            raise ValueError(f"Unknown rounding method: {method}")
    
    def create_qhdopt_model(self, M: np.ndarray, bounds: Tuple[float, float] = (0, 1)) -> QHD:
        """
        Create QHDOPT model from BoxQP matrix.
        
        BoxQP problem: min x^T M x subject to 0 <= x <= 1
        QHDOPT expects: min (1/2) x^T Q x + b^T x
        So we need: Q = 2*M, b = 0
        
        Args:
            M: The quadratic matrix from BoxQP
            bounds: Box constraints (default: (0, 1))
            
        Returns:
            QHD model instance
        """
        Q = 2 * M  
        b = np.zeros(M.shape[0]) 
        
        Q_list = Q.tolist()
        b_list = b.tolist()
        
        qhdopt_model = QHD.QP(Q_list, b_list, bounds=bounds)
        
        return qhdopt_model
    
    def solve_instance(self, n: int, p: float, N: int, idx: int, 
                      backend: str = "dwave_sim",   
                      resolution: int = 8,
                      shots: int = 100,
                      verbose: int = 1,
                      **backend_kwargs) -> dict:
        """
        Solve a specific BoxQP instance using QHDOPT.
        
        Args:
            n, p, N, idx: Instance parameters
            backend: Backend to use ("dwave", "qutip", "ionq", "dwave_sim")
            resolution: Resolution for quantum encoding
            shots: Number of shots/samples
            verbose: Verbosity level
            **backend_kwargs: Additional backend-specific parameters
            
        Returns:
            Dictionary containing results
        """
        M, reference_solution = self.load_instance(n, p, N, idx)
        
        # Create QHDOPT model
        model = self.create_qhdopt_model(M)
        
        # Setup backend
        if backend == "dwave":
            model.dwave_setup(resolution=resolution, shots=shots, **backend_kwargs)
        elif backend == "qutip":
            model.qutip_setup(resolution=resolution, shots=shots, **backend_kwargs)
        elif backend == "ionq":
            ionq_config = {
                'resolution': resolution,
                'shots': shots,
                'api_key_from_file': "ionqapi.txt",
                'embedding_scheme': "onehot",
                'post_processing_method': "TNC",
                'on_simulator': True,
                'backend': 'harmony'
            }
            ionq_config.update(backend_kwargs)
            model.ionq_setup(**ionq_config)
        elif backend == "dwave_sim":
            model.dwave_sim_setup(
                resolution=resolution,
                shots=shots,
                post_processing_method="TNC",
                **backend_kwargs
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        
        # Solve
        response = model.optimize(verbose=verbose)
        
        # Get solutions
        refined_solution = response.refined_minimizer if response.refined_minimizer is not None else response.minimizer
        unrefined_solution = response.minimizer
        
        # Calculate objectives (like summary.csv format: x^T M x)
        refined_obj = float(refined_solution.T @ M @ refined_solution)
        unrefined_obj = float(unrefined_solution.T @ M @ unrefined_solution)
        
        # Extract timing information from response.info
        info = response.info
        compile_time = info.get('compile_time', 0.0)
        backend_time = info.get('backend_time', 0.0)
        decoding_time = info.get('decoding_time', 0.0)
        refining_time = info.get('refining_time', 0.0) if info.get('refine_status', False) else 0.0
        total_runtime = compile_time + backend_time + decoding_time + refining_time
        
        return {
            'n': n, 'p': p, 'N': N, 'idx': idx,
            'backend': backend,
            'resolution': resolution,
            'shots': shots,

            # Solutions
            'refined_solution': refined_solution,
            'unrefined_solution': unrefined_solution,
            
            # Objectives (calculated as x^T M x like summary.csv)
            'refined_objective': refined_obj,
            'unrefined_objective': unrefined_obj,
            
            # QHDOPT raw results
            'qhdopt_refined_minimum': response.refined_minimum,
            'qhdopt_unrefined_minimum': response.minimum,
            
            # Store the response object for batch processing
            'response': response,
            
            # Timing information
            'total_runtime': total_runtime,
            'compile_time': compile_time,
            'backend_time': backend_time,
            'decoding_time': decoding_time,
            'refining_time': refining_time,
            
            'success': True
        }

    def solve_batch(self, n_values, p_values, N_values, idx_values, backend="dwave_sim", resolution=8, shots=100, verbose=1, **backend_kwargs):
        """
        Solve instances for all specified parameter combinations and output results to CSV.
        Columns: n, p, N, idx, objVal, runtime(s) (matching summary.csv format)
        """
        os.makedirs("Results", exist_ok=True)
        import time
        results = []
        total_instances = len(n_values) * len(p_values) * len(N_values) * len(idx_values)
        processed = 0
        for n in n_values:
            for p in p_values:
                for N in N_values:
                    for idx in idx_values:
                        try:
                            result = self.solve_instance(
                                n=n, p=p, N=N, idx=idx,
                                backend=backend, resolution=resolution, shots=shots, verbose=0, **backend_kwargs
                            )
                            # Extract values directly from result dict
                            refined_obj = result.get('refined_objective', None)
                            total_runtime = result.get('total_runtime', None)
                            backend_time = result.get('backend_time', None)
                            
                            row = {
                                'n': n,
                                'p': p,
                                'N': N,
                                'idx': idx,
                                'objVal': refined_obj,  # Match summary.csv column name
                                'runtime(s)': total_runtime,  # Match summary.csv column name
                            }
                            results.append(row)
                            processed += 1
                            status = "✓" if refined_obj is not None else "✗"
                            print(f"Progress: {processed}/{total_instances} - n={n}, p={p}, N={N}, idx={idx}: {status} | objVal={refined_obj} | runtime={total_runtime:.6f}s")
                        except FileNotFoundError:
                            processed += 1
                            print(f"Progress: {processed}/{total_instances} - n={n}, p={p}, N={N}, idx={idx}: skipped (matrix not found)")
                            continue
                        except Exception as e:
                            processed += 1
                            print(f"Progress: {processed}/{total_instances} - n={n}, p={p}, N={N}, idx={idx}: ✗ (Error: {e})")
                            continue
        # Save to CSV
        if results:
            df = pd.DataFrame(results)
            cols = ['n', 'p', 'N', 'idx', 'objVal', 'runtime(s)']  # Match summary.csv format
            df = df[cols]
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"Results/qhdopt_batch_n={n}_{timestamp}.csv"
            df.to_csv(output_file, index=False)
            print(f"\nResults saved to {output_file}")
            # Print summary
            success_count = df['objVal'].notna().sum()
            print(f"\nSummary:")
            print(f"Total instances: {len(df)}")
            print(f"Successful: {success_count}")
            print(f"Failed: {len(df) - success_count}")
            if success_count > 0:
                avg_runtime = df['runtime(s)'].dropna().mean()
                print(f"Average runtime: {avg_runtime:.3f}s")





def main():
    solver = BoxQPSolver()
    # Example: Solve a batch of instances
    solver.solve_batch(
        n_values=[10],
        p_values=[0.25, 0.5],
        N_values=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
        idx_values=[1,2,3],
        backend="dwave_sim",
        resolution=8,
        shots=100,
        verbose=1
    )
    
    # Example: Solve a single instance
    """
    result = solver.solve_instance(
            n=10, p=0.25, N=1, idx=2,
            backend="dwave_sim",
            resolution=8,
            shots=100,
            verbose=1,
        )
        
    if result['success']:
        print(f"\nQHDOPT:")
        print(f"  Refined: {result['refined_objective']:.6f}")
        print(f"  Unrefined: {result['unrefined_objective']:.6f}")

    else:
        print(f"QHDOPT failed")
    """
    

if __name__ == "__main__":
    main()