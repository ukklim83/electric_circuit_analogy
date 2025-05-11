# electric_circuit_analogy

## Code Structure
- Consists of `electric_analogy_programming.py` and `electric_analogy.py`.

- `electric_analogy_programming.py`: Sets up channel geometry and configuration, executes functions

- `electric_analogy.py`: Shared across all channels, calculates flow rate/concentration and optimizes channel length

## `electric_analogy_programming.py`
- Imports geometric structure for each channel from CSV files
    - `incidence_mat.csv`
    - `length_mat.csv`
    - `concentration_mat.csv`

- Sets inlet conditions for each channel
    - `ini_list`
        - `["b or f", node, value, "i or o", m^3/s or Pa]`

- Executes functions from `electric_analogy.py`
    - `execute_functions()`: Calculates flow rate and concentration
    - `execute_length_change()`: Optimizes channel length
    - `plotting_outlet()`: Plots outlet flow rate and concentration

## `electric_analogy.py`
- Function execution sequence
    - `execute_functions()`
        - Variable setup
        - Imports incidence_mat & length_mat
        - Creates block matrices
            - Resistance calculations for various channel types are imported from `resistance.py`
        - `Kirchhoff_solver()`: Calculates flow rate using **Kirchhoff's law**
        - `conc_calculation()`: Calculates concentration using modified **Kirchhoff's current law**
        
    - `plotting_outlet()`: Plots calculated flow rates and concentrations
    
    - `execute_length_change()`
        - Calculates constraints for flow rate and pressure
        - `MicrofluidicOptimizationProblem()`: Defines **multi-objective optimization** problem
        - Plots and saves Pareto front
        - Saves newly obtained revised length as `new_length_mat.csv`
        - Runs `execute_functions()` with `new_length_mat.csv` to verify improved results

- Key function descriptions
    - `Kirchhoff_solver()`
        - Divides modes based on whether there are unknowns in vector $\mathbf{f}$ (`isSourceUnknown`) and whether inlet conditions are given as flow rate or pressure (`Q_driven` or `P_driven`)
        - `isSourceUnknown == False`
            - Generally solves $\left[\begin{array}{cc}R&A\A^T&0\end{array}\right]\left[\begin{array}{c}q\p\right]=\left[\begin{array}{c}b\f\end{array}\right]$, which requires **grounding** by setting one outlet pressure value to 0
            - For `Q_driven`, directly solves the equation (`solve_current_driven()`)
            - For `P_driven`, rearranges vectors $\mathbf{p}_i$ and $\mathbf{f}_i$ (`solve_pressure_driven()`)
        
        - `isSourceUnknown == True`
            - Includes $\mathbf{p}_o = 0$ in the equation, so no additional **grounding** is needed
            - Both cases appropriately rearrange and solve equations (`solve_unknown_source_current_driven()`, `solve_unknown_source_pressure_driven()`)
            
        - For all `P_driven` cases, returns modified `b_f` with calculated inlet flow rates (for concentration calculations)
        
    - `conc_calculation()`
        - Constructs Modified Kirchhoff's current law equations
            - Assigns diagonal matrices variables for $\mathbf{q}$ & $\mathbf{f}$ vectors (`i_mat`, `f_mat`)
            - Uses `mat_divide()` to split incidence matrix, flow rate, and source vector into inlet/intermediate/outlet parts
            - Specifies additional constraints for **diverging nodes** in `two_or_more_outlet_mats`
            - Combines these to form left and right block matrices
            
        - Derives and solves normal equation since left-hand matrix is not square
        - Returns outlet concentration

    - `execute_length_change()`
        - Reads desired concentration from `concentration_mat.csv` to calculate target concentration
        - Calculates target flow as equal distribution of total inlet flow to each outlet (`calculate_currents()`)
            - If inlet condition is given as pressure, calculates inlet flow rate at that pressure using Kirchhoff solver once (`calculate_pressure_driven_current`)
            
        - `MicrofluidicOptimizationProblem()`: Uses *pymoo* module which defines problems in Problem class, creating an object with `problem = Problem()` as an argument for minimize operation
            - Includes constraints that flow rate and concentration must be non-negative to filter physically impossible cases
            - Sets objective function as the **norm** of the **difference** between target and actual flow rates and concentrations
            - `constraint_diff()` calculates this difference
                - Returns the difference between target flow/concentration and actual flow/concentration calculated by `length_change_func()`
                - Target and actual flow rates are scaled by total inlet flow  
            - All functions are vectorized for faster computation
                - pymoo `Problem()` creates a single numpy array (`shape = popSize * n_var`) with samples of the specified population size (`popSize`)
                - Instead of using `for i in popSize` loops which slow down execution
                - Functions are vectorized to process `popSize * n_var` sized variables at once:
                    - [] `construct_block_mat_vec()`
                    - [] `ground_block_mat_vec()`
                    - [] `Kirchhoff_solver_vec()`
                    - [] `solve_current_driven_vec()`
                    - [] `solve_pressure_driven_vec()`
                    - [] `solve_unknown_source_current_driven_vec()`
                    - [] `solve_unknown_source_pressure_driven_vec()`
                    - [] `ground_calculation_vec()`
                    - [] `ground_off_vec()`
                    - [] `conc_calculation_vec()`
                    - [] `mat_divide_vec()`

        - Defines algorithm and callback for optimization monitoring, then runs optimization
        - Plots Pareto front (`process_optimization_results`)
        - Saves improved length to CSV file