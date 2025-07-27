import os
import time
import numpy as np
import scipy.sparse as sp
import pandas as pd
from typing import List, Optional, Dict, Any
import gurobipy as gp
from gurobipy import GRB

class BoxQPGurobiSolver:
    
    """ Problem format: min x^T M x subject to 0 <= x <= 1
    """
    
    def __init__(self, data_dir: str = "BoxQP_benchmarks-main/max_clique", silent: bool = True):
        self.data_dir = data_dir
        self.silent = silent
        os.makedirs("Results", exist_ok=True)
    
    def load_matrix(self, n: int, p: float, N: int, idx: int) -> np.ndarray:
        """Load sparse matrix M from .npz file."""
        matrix_filename = f"M_n={n}_p={p}_N={N}_idx={idx}.npz"
        matrix_path = os.path.join(self.data_dir, matrix_filename)
        
        if not os.path.exists(matrix_path):
            raise FileNotFoundError(f"Matrix file not found: {matrix_path}")
        
        return sp.load_npz(matrix_path).toarray()

    def solve_instance(self, n: int, p: float, N: int, idx: int, 
                      fixed_vars: Optional[Dict[int, float]] = None) -> Dict[str, Any]:
        """
        Solve BoxQP instance with optional variable fixing.
        
        Args:
            n, p, N, idx: Instance parameters
            fixed_vars: Dict mapping variable indices to fixed values {i: val}
        """
        try:
            M = self.load_matrix(n, p, N, idx)
            
            with gp.Env(empty=True) as env:
                env.setParam('OutputFlag', 0)
                env.start()
                
                with gp.Model(env=env) as model:
                    x = model.addMVar(shape=n, lb=0.0, ub=1.0, name="x")
                    
                    # Apply variable fixing if provided
                    if fixed_vars:
                        for i, val in fixed_vars.items():
                            if 0 <= i < n:
                                x[i].lb = val
                                x[i].ub = val
                    
                    model.setObjective(x @ M @ x, GRB.MINIMIZE)
                    model.optimize()
                    
                    if model.status == GRB.OPTIMAL:
                        solution = x.X
                        return {
                            'n': n, 'p': p, 'N': N, 'idx': idx,
                            'objVal': float(model.objVal),
                            'runtime(s)': model.Runtime,
                            'solution': solution,
                            'status': 'optimal',
                            'fixed_vars': fixed_vars or {}
                        }
                    else:
                        return {
                            'n': n, 'p': p, 'N': N, 'idx': idx,
                            'objVal': None, 'runtime(s)': None,
                            'solution': None, 'status': 'failed',
                            'fixed_vars': fixed_vars or {}
                        }
            
        except FileNotFoundError:
            raise  
        except Exception as e:
            return {
                'n': n, 'p': p, 'N': N, 'idx': idx,
                'objVal': None, 'runtime(s)': None,
                'solution': None, 'status': f'error: {str(e)}',
                'fixed_vars': fixed_vars or {}
            }

    def solve_batch(self, n_values: List[int], p_values: List[float], 
                   N_values: List[int], idx_values: List[int], 
                   verbose: bool = False) -> pd.DataFrame:
        """Solve instances for all specified parameter combinations."""
        results = []
        total_instances = len(n_values) * len(p_values) * len(N_values) * len(idx_values)
        processed = 0
        
        for n in n_values:
            for p in p_values:
                for N in N_values:
                    for idx in idx_values:
                        try:
                            result = self.solve_instance(n=n, p=p, N=N, idx=idx)
                            results.append(result)
                            processed += 1
                            
                            if verbose:
                                status = "✓" if result['objVal'] is not None else "✗"
                                obj_str = f"{result['objVal']:.6f}" if result['objVal'] is not None else "None"
                                print(f"{processed}/{total_instances} - n={n}, p={p}, N={N}, idx={idx}: {status} | objVal={obj_str}")
                                
                        except FileNotFoundError:
                            processed += 1
                            if verbose:
                                print(f"{processed}/{total_instances} - n={n}, p={p}, N={N}, idx={idx}: skipped (not found)")
                            continue
        
        # Save results
        if results:
            df = pd.DataFrame(results)
            df = df[['n', 'p', 'N', 'idx', 'objVal', 'runtime(s)']]
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"Results/gurobi_continuous_{timestamp}.csv"
            df.to_csv(output_file, index=False)
            
            if verbose:
                success_count = df['objVal'].notna().sum()
                print(f"\nResults saved to {output_file}")
                print(f"Summary: {success_count}/{len(df)} successful, avg runtime: {df['runtime(s)'].mean():.3f}s")
            
            return df
        
        return pd.DataFrame()

def main():
    solver = BoxQPGurobiSolver()
    
    # Solve batch silently
    df = solver.solve_batch(
        n_values=[10],
        p_values=[0.25],
        N_values=list(range(1,10)),
        idx_values=[1, 2],
        verbose=True
    )
    
    print(f" Completed {len(df)} instances")

if __name__ == "__main__":
    main()
