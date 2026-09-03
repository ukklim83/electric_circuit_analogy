import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as optimize
import copy
from pymoo.core.problem import Problem
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.core.callback import Callback
from pymoo.visualization.scatter import Scatter
from scipy.spatial.distance import cdist
import time
import sys
try:
    from .resistance import *
except ImportError:  # Support direct execution from the codes directory.
    from resistance import *


class MicrofluidicOptimizationProblem(Problem):
    """
    A class representing the microfluidic optimization problem.

    This class inherits from the Problem class and defines the specific optimization
    problem for microfluidic systems.
    """

    def __init__(
        self,
        constraint,
        whatToSolve,
        chainging_edges,
        inc_csv,
        length_csv,
        args,
        ini_list,
        inlet_node_idx,
        inlet_edge_idx,
        outlet_node_idx,
        outlet_edge_idx,
        address,
        lb,
        ub,
        flow_wght,
        conc_wght,
        popSize,
    ):
        """
        Initialize the MicrofluidicOptimizationProblem.

        Args:
            constraint (list): Constraints for the optimization problem.
            whatToSolve (str or list): Indicates what to solve in the problem.
            changing_edges (list): List of edges that can be changed.
            inc_csv (str): Filename of the incidence matrix CSV.
            length_csv (str): Filename of the length matrix CSV.
            conc_csv (str): Filename of the concentration matrix CSV.
            args (tuple): Additional arguments for the problem.
            ini_list (list): Initial list of values.
            inlet_node_idx (list): Indices of inlet nodes.
            inlet_edge_idx (list): Indices of inlet edges.
            outlet_node_idx (list): Indices of outlet nodes.
            outlet_edge_idx (list): Indices of outlet edges.
            address (str): File path for saving results.
            lb (list): Lower bounds for variables.
            ub (list): Upper bounds for variables.
            flow_wght (float): Weight for flow in the objective function.
            conc_wght (float): Weight for concentration in the objective function.
            popSize (int): Population size for the optimization algorithm.
        """
        super().__init__(
            n_var=len(chainging_edges),
            n_obj=2,
            n_constr=2,
            xl=lb,
            xu=ub,
        )
        self.constraint = constraint
        self.whatToSolve = whatToSolve
        self.chainging_edges = chainging_edges
        self.inc_csv = inc_csv
        self.length_csv = length_csv
        self.args = args
        self.ini_list = ini_list
        self.inlet_node_idx = inlet_node_idx
        self.inlet_edge_idx = inlet_edge_idx
        self.outlet_node_idx = outlet_node_idx
        self.outlet_edge_idx = outlet_edge_idx
        self.address = address
        self.threshold = 1e-5
        self.flow_wght = flow_wght
        self.conc_wght = conc_wght
        self.popSize = popSize

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate the optimization problem.

        This method calculates the objective functions and constraints for given decision variables.

        Args:
            x (numpy.ndarray): Decision variables.
            out (dict): Dictionary to store the output values.

        Returns:
            None
        """
        out_F = []
        out_G = []
        flow_diff, conc_diff, flow, conc = constraint_diff(
            self.constraint,
            x,
            self.whatToSolve,
            self.chainging_edges,
            self.inc_csv,
            self.length_csv,
            self.args,
            self.ini_list,
            self.inlet_node_idx,
            self.inlet_edge_idx,
            self.address,
            self.popSize,
        )

        outlet_edges = self.outlet_edge_idx

        print(flow)
        print(conc)

        flow_cond = np.max(-1 * flow[:, : len(outlet_edges)], axis=1) - self.threshold
        conc_cond = np.max(-1 * conc, axis=1) - self.threshold

        flow_norm = np.linalg.norm(flow_diff, axis=1) * self.flow_wght
        conc_norm = np.linalg.norm(conc_diff, axis=1) * self.conc_wght

        out_F = np.vstack((flow_norm, conc_norm)).T
        out_G = np.vstack((flow_cond, conc_cond)).T

        print(out_F)
        print(out_G)

        out["F"] = out_F
        out["G"] = out_G


class MicrofluidicOptimalpointFinding(ElementwiseProblem):
    """
    A class for finding optimal points in the microfluidic system.

    This class inherits from ElementwiseProblem and is used to find optimal points
    in the microfluidic system based on specific criteria.
    """

    def __init__(
        self,
        whatToSolve,
        changing_edges,
        inc_csv,
        length_csv,
        conc_csv,
        args,
        ini_list,
        inlet_node_idx,
        inlet_edge_idx,
        outlet_node_idx,
        outlet_edge_idx,
        conc_wght,
        flow_wght,
        address,
        lb,
        ub,
    ):
        """
        Initialize the MicrofluidicOptimalpointFinding problem.

        Args:
            whatToSolve (str or list): Indicates what to solve in the problem.
            changing_edges (list): List of edges that can be changed.
            inc_csv (str): Filename of the incidence matrix CSV.
            length_csv (str): Filename of the length matrix CSV.
            conc_csv (str): Filename of the concentration matrix CSV.
            args (tuple): Additional arguments for the problem.
            ini_list (list): Initial list of values.
            inlet_node_idx (list): Indices of inlet nodes.
            inlet_edge_idx (list): Indices of inlet edges.
            conc_wght (float): Weight for concentration in the objective function.
            flow_wght (float): Weight for flow in the objective function.
            address (str): File path for saving results.
            lb (list): Lower bounds for variables.
            ub (list): Upper bounds for variables.
        """
        super().__init__(n_var=2, n_obj=3, n_constr=0, xl=lb, xu=ub)
        self.whatToSolve = whatToSolve
        self.changing_edges = changing_edges
        self.inc_csv = inc_csv
        self.length_csv = length_csv
        self.conc_csv = conc_csv
        self.args = args
        self.ini_list = ini_list
        self.inlet_node_idx = inlet_node_idx
        self.inlet_edge_idx = inlet_edge_idx
        self.outlet_node_idx = outlet_node_idx
        self.outlet_edge_idx = outlet_edge_idx
        self.conc_wght = conc_wght
        self.flow_wght = flow_wght
        self.address = address

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate the optimal point finding problem.

        This method calculates the objective functions for given decision variables.

        Args:
            x (numpy.ndarray): Decision variables.
            out (dict): Dictionary to store the output values.

        Returns:
            None
        """
        pop_size = x[0]
        n_gen = x[1]
        res = opt_time_acc(
            self.whatToSolve,
            self.changing_edges,
            self.inc_csv,
            self.length_csv,
            self.conc_csv,
            self.args,
            self.ini_list,
            self.inlet_node_idx,
            self.inlet_edge_idx,
            self.outlet_node_idx,
            self.outlet_edge_idx,
            self.conc_wght,
            self.flow_wght,
            self.address,
            pop_size,
            n_gen,
        )

        flow_diff = res[0][0]
        conc_diff = res[0][1]
        time_diff = res[1]

        out["F"] = [flow_diff, conc_diff, time_diff]
        print(out["F"])


# Define the callback
class ConvergenceCallback(Callback):
    """
    A callback class to track convergence during optimization.

    This class is used to collect and store data about the optimization process,
    including the number of evaluations, optimal solutions, and population data.
    """

    def __init__(self) -> None:
        """
        Initialize the ConvergenceCallback.

        Initializes empty lists to store optimization data.
        """
        super().__init__()
        self.n_evals = []
        self.opt = []
        self.data = {"X": [], "F": []}

    def notify(self, algorithm):
        """
        Collect data at each iteration of the optimization algorithm.

        This method is called automatically by the optimization algorithm
        at each iteration to update the callback data.

        Args:
            algorithm: The optimization algorithm object.

        Returns:
            None
        """
        self.n_evals.append(algorithm.evaluator.n_eval)
        self.opt.append(algorithm.opt.get("F"))
        self.data["X"].append(algorithm.pop.get("X"))
        self.data["F"].append(algorithm.pop.get("F"))


def variable_setup(whatToSolve):
    """
    Set up boolean flags for different variables to be solved in the problem.

    This function determines which variables are unknown based on the input
    'whatToSolve' parameter. It sets boolean flags for length, current, voltage,
    source, and concentration variables.

    Args:
        whatToSolve (str or list): Indicates what to solve in the problem.
            Can be a single string or a list of strings.
            Possible values: 'length', 'current', 'voltage', 'source', 'concentration'.

    Returns:
        tuple: A tuple of boolean flags in the following order:
            (isLengthUnknown, isSourceUnknown, isConcUnknown)
    """
    isLengthUnknown = False
    isSourceUnknown = False
    isConcUnknown = False

    if type(whatToSolve) == str:
        if whatToSolve == "length":
            isLengthUnknown = True
        elif whatToSolve == "source":
            isSourceUnknown = True
        elif whatToSolve == "concentration":
            isConcUnknown = True

    elif type(whatToSolve) == list:
        if "length" in whatToSolve:
            isLengthUnknown = True
        if "source" in whatToSolve:
            isSourceUnknown = True
        if "concentration" in whatToSolve:
            isConcUnknown = True

    return (
        isLengthUnknown,
        isSourceUnknown,
        isConcUnknown,
    )


def inc_mat_produce(inc_csv, address):
    """
    Produce incidence matrix from CSV file.

    This function reads an incidence matrix from a CSV file and returns it as both
    a pandas DataFrame and a numpy array.

    Args:
        inc_csv (str): Filename of the incidence matrix CSV.
        address (str): File path for the CSV.

    Returns:
        inc_mat (np.ndarray): Numpy array of the incidence matrix.
    """
    # Read the CSV file into a pandas DataFrame
    inc_df = pd.read_csv(f"{address}/{inc_csv}", index_col="edge")

    # Fill any NaN values with 0
    inc_df = inc_df.fillna(0)

    # Convert the DataFrame to a numpy array
    inc_mat = inc_df.to_numpy()

    return inc_mat


def length_mat_produce(length_csv, address):
    """
    Produce length matrix from CSV file.

    This function reads a CSV file containing length data and returns it as a pandas DataFrame.

    Args:
        length_csv (str): Filename of the length matrix CSV.
        address (str): File path for the CSV.

    Returns:
        pd.DataFrame: DataFrame of the length matrix with 'edge' as the index.

    Raises:
        FileNotFoundError: If the specified CSV file is not found at the given address.
        pd.errors.EmptyDataError: If the CSV file is empty.
    """
    try:
        length_df = pd.read_csv(f"{address}/{length_csv}", index_col="edge")

        if length_df.empty:
            raise pd.errors.EmptyDataError("The length matrix CSV file is empty.")

        return length_df

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Length matrix CSV file not found at {address}/{length_csv}"
        )
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(
            "The length matrix CSV file is empty or contains no valid data."
        )
    except Exception as e:
        raise Exception(
            f"An error occurred while reading the length matrix CSV: {str(e)}"
        )


def conc_mat_produce(conc_csv, address):
    """
    Produce concentration matrix from CSV file.

    This function reads a CSV file containing concentration data and returns it as a pandas DataFrame.

    Args:
        conc_csv (str): Filename of the concentration matrix CSV.
        address (str): File path for the CSV.

    Returns:
        pd.DataFrame: DataFrame of the concentration matrix with 'outlets' as the index.

    Raises:
        FileNotFoundError: If the specified CSV file is not found at the given address.
        pd.errors.EmptyDataError: If the CSV file is empty or contains no valid data.
    """
    try:
        conc_df = pd.read_csv(f"{address}/{conc_csv}", index_col="outlets")

        if conc_df.empty:
            raise pd.errors.EmptyDataError(
                "The concentration matrix CSV file is empty."
            )

        return conc_df

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Concentration matrix CSV file not found at {address}/{conc_csv}"
        )
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(
            "The concentration matrix CSV file is empty or contains no valid data."
        )
    except Exception as e:
        raise Exception(
            f"An error occurred while reading the concentration matrix CSV: {str(e)}"
        )


def R_H(eta, L, args, res_type="rectangular"):
    """
    Compute the hydraulic resistance (R_H) for different channel cross-sections.

    This function calculates the hydraulic resistance of microfluidic channels
    based on their geometry and fluid properties.

    Args:
        eta (float): Fluid viscosity (SI units, Pa·s).
        L (np.array): Array of channel lengths (m).
        args (tuple): Channel geometry parameters, depending on res_type.
        res_type (str, optional): Channel cross-section type.
            Options: "rectangular", "ellipse", "triangular", "harmonical_perturbed_circle".
            Defaults to "rectangular".

    Returns:
        np.array: Hydraulic resistance matrix (R_H) for the given channels.

    Raises:
        ValueError: If an invalid res_type is provided.
    """
    if res_type == "rectangular":
        w, h, n = args
        alpha, area = rectangular(w, h, n)

    elif res_type == "ellipse":
        a, b = args
        alpha, area = ellipse(a, b)

    elif res_type == "triangular":
        a, b, c = args
        alpha, area = triangular(a, b, c)

    elif res_type == "harmonical_perturbed_circle":
        epsilon, k = args
        alpha, area = harmonical_perturbed_circle(epsilon, k)

    else:
        raise ValueError(f"Invalid res_type: {res_type}")

    R_H_ideal = eta * L / (area**2)
    return alpha * R_H_ideal


def construct_block_mat(length, eta, args, inc_mat):
    """
    summary:
        function to combine resistance matrix, incidence matrix to construct block matrix

    Args:
        length (list): the list of length matrix
        eta (float): viscosity
        w (float): channel width
        h (float): channel height
        n (int): appropriate coefficient to compute series sum
        inc_mat (np.array): incidence matrix numpy array

    Returns:
        L (np.array): numpy array consists of the lengths of the each channels (length matrix)
        resist_mat (np.array): diagonal matrix filled with each resistances.
        A1 (np.array): horizontally combine R and A matrix.
        A2 (np.array): horizontally combine A^T and 0 matrix.
        A (np.array): construct block matrix to solve problem.
    """
    L = np.diag(
        [length[i] * 1e-3 for i in range(inc_mat.shape[0])]
    )  # matrix of the length of channels for deriving resistance

    # Calculate resistance matrix
    resist_mat = R_H(eta, L, args, "rectangular")  # matrix of resistance

    # Construct upper and lower halves of the block matrix
    A1 = np.hstack((resist_mat, inc_mat))
    A2 = np.hstack((inc_mat.T, np.zeros((inc_mat.shape[1], inc_mat.shape[1]))))

    # Combine to form the complete block matrix
    A = np.vstack((A1, A2))

    return A


def construct_block_mat_vec(length_vec, eta, args, inc_mat_vec, popSize):
    """
    Construct block matrices for vectorized calculations.

    This function combines resistance matrices and incidence matrices to create
    block matrices used in solving the Kirchhoff's equations for multiple microfluidic systems.

    Args:
        length_vec (np.ndarray): Array of channel lengths for multiple systems (shape: popSize x num_channels).
        eta (float): Fluid viscosity (in Pa·s).
        args (tuple): Contains channel width (w), height (h), and computation coefficient (n).
        inc_mat_vec (np.ndarray): Incidence matrices for multiple systems (shape: popSize x num_nodes x num_channels).
        popSize (int): Number of systems to process in parallel.

    Returns:
        A_vec (np.ndarray): Complete block matrices to solve the problems for each system.
    """
    # Convert lengths from mm to m and create diagonal matrix
    L_vec = (
        np.where(np.eye(inc_mat_vec.shape[1], dtype=bool), length_vec[:, np.newaxis], 0)
        * 1e-3
    )  # matrix of the length of channels for deriving resistance

    # Calculate resistance matrices
    resist_mat_vec = R_H(eta, L_vec, args, "rectangular")  # matrix of resistance

    # Construct upper and lower halves of the block matrices
    A1_vec = np.concatenate((resist_mat_vec, inc_mat_vec), axis=2)
    A2_vec = np.concatenate(
        (
            np.transpose(inc_mat_vec, axes=(0, 2, 1)),
            np.zeros((popSize, inc_mat_vec.shape[2], inc_mat_vec.shape[2])),
        ),
        axis=2,
    )

    # Combine to form the complete block matrices
    A_vec = np.concatenate((A1_vec, A2_vec), axis=1)

    return A_vec


def construct_RHS(inc_mat, ini_list):
    """
    Construct the right-hand side (RHS) vector for the Kirchhoff's circuit equations.

    This function creates the b and f vectors, which represent voltage and current sources
    respectively, and combines them into a single b_f vector.

    Args:
        inc_mat (np.ndarray): Incidence matrix of the circuit.
        ini_list (list): List of initial values for voltage and current sources.
            Each element is a list [type, index, value, location, unit], where:
            - type: 'b' for voltage source, 'f' for current source
            - index: Integer index for the source
            - value: Numerical value of the source
            - location: 'o' for outlet, 'i' for inlet
            - unit: Unit of measurement (e.g., 'Pa' for pressure, 'm^3/s' for flow rate)

    Returns:
        b_f (np.ndarray): Combined vector of voltage and current sources.
    """
    b = np.zeros(inc_mat.shape[0])
    f = np.zeros(inc_mat.shape[1])

    for item in ini_list:
        if item[0] == "b":  # Voltage source
            b[item[1]] = item[2]
        elif item[0] == "f":  # Current source
            f[item[1]] = item[2]

    b_f = np.hstack((b, f))
    return b_f


def ground_block_mat(A, inc_mat, b_f, ini_list_outlet):
    """
    Construct the ground block matrix to solve the problem.

    This function modifies the input matrix and vector by removing a ground block,
    which is necessary for solving electrical circuit problems.

    Args:
        A (numpy.ndarray): The input matrix representing the circuit equations.
        inc_mat (numpy.ndarray): The incidence matrix of the circuit.
        b_f (numpy.ndarray): The input vector representing the right-hand side of the equations.
        ini_list_outlet (list): The list of initial values for the outlets.

    Returns:
        tuple: Contains the following:
            A_ground (numpy.ndarray): The modified matrix after removing the ground block.
            b_f_ground (numpy.ndarray): The modified vector after removing the ground block.
            gnd_idx (int): The index of the ground block that was removed.
    """
    # Randomly select a ground node from the outlets
    gnd_idx = np.random.randint(0, len(ini_list_outlet))

    # Remove the corresponding row and column from A
    A_ground = np.delete(A, inc_mat.shape[0] + gnd_idx, axis=1)
    A_ground = np.delete(A_ground, inc_mat.shape[0] + gnd_idx, axis=0)

    # Remove the corresponding element from b_f
    b_f_ground = np.delete(b_f, inc_mat.shape[0] + gnd_idx)

    return A_ground, b_f_ground, gnd_idx


def ground_block_mat_vec(A_vec, inc_mat_vec, b_f_vec, ini_list_outlet):
    """
    Construct the ground block matrices for vectorized calculations.

    This function modifies the input matrices and vectors by removing a ground block,
    which is necessary for solving electrical circuit problems for multiple systems simultaneously.

    Args:
        A_vec (np.ndarray): The input matrices representing the circuit equations for multiple systems.
        inc_mat_vec (np.ndarray): The incidence matrices of the circuits for multiple systems.
        b_f_vec (np.ndarray): The input vectors representing the right-hand side of the equations for multiple systems.
        ini_list_outlet (list): The list of initial values for the outlets.

    Returns:
        tuple: Contains the following:
            A_ground_vec (np.ndarray): The modified matrices after removing the ground block.
            b_f_ground_vec (np.ndarray): The modified vectors after removing the ground block.
            gnd_idx (int): The index of the ground block that was removed.
    """
    # Randomly select a ground node from the outlets
    gnd_idx = np.random.randint(0, len(ini_list_outlet))

    # Remove the corresponding row and column from A_vec
    A_ground_vec = np.delete(A_vec, inc_mat_vec.shape[1] + gnd_idx, axis=2)
    A_ground_vec = np.delete(A_ground_vec, inc_mat_vec.shape[1] + gnd_idx, axis=1)

    # Remove the corresponding element from b_f_vec
    b_f_ground_vec = np.delete(b_f_vec, inc_mat_vec.shape[1] + gnd_idx, axis=1)

    return A_ground_vec, b_f_ground_vec, gnd_idx


def Kirchhoff_solver(
    A,
    A_ground,
    b_f,
    b_f_ground,
    inc_mat,
    isSourceUnknown,
    ini_list,
    ini_list_outlet,
    ini_list_inlet,
    gnd_idx,
):
    """
    Solves the Kirchhoff's circuit equations for a given microfluidic circuit.

    This function handles both current-driven and pressure-driven scenarios,
    as well as cases where the source is known or unknown.

    Args:
        A (np.ndarray): The constructed block matrix to solve the problem.
        A_ground (np.ndarray): The ground matrix for A.
        b_f (np.ndarray): The combined vector of voltage and current sources.
        b_f_ground (np.ndarray): The ground vector for b_f.
        inc_mat (np.ndarray): The incidence matrix of the circuit.
        isSourceUnknown (bool): True if the source is unknown, False otherwise.
        ini_list (list): Initial values of voltage & current sources.
        ini_list_outlet (list): The ini_list only with outlet streams.
        ini_list_inlet (list): The ini_list only with inlet streams.
        gnd_idx (int): The index of the ground node.

    Returns:
        tuple: Contains the following:
            i_vec (np.ndarray): The current vector obtained from x (in mL/min).
            p_vec (np.ndarray): The pressure vector obtained from x.
            b_f (np.ndarray): The updated b_f vector.
    """
    # Determine if the system is current-driven or pressure-driven
    mode = next((i[4] for i in ini_list if i[3] == "i"), None)
    mode = "Q_driven" if mode == "m^3/s" else "P_driven"

    if not isSourceUnknown:
        if mode == "Q_driven":
            return solve_current_driven(A_ground, b_f_ground, inc_mat, gnd_idx, b_f)

        elif mode == "P_driven":
            return solve_pressure_driven(
                A_ground,
                b_f_ground,
                inc_mat,
                ini_list_inlet,
                ini_list_outlet,
                gnd_idx,
                b_f,
            )

    else:
        if mode == "Q_driven":
            return solve_unknown_source_current_driven(
                A, b_f, inc_mat, ini_list, ini_list_outlet, ini_list_inlet
            )

        elif mode == "P_driven":
            return solve_unknown_source_pressure_driven(
                A, b_f, inc_mat, ini_list, ini_list_outlet, ini_list_inlet
            )


def solve_current_driven(A_ground, b_f_ground, inc_mat, gnd_idx, b_f):
    """
    Solves the current-driven case with known source.

    Args:
        A_ground (np.ndarray): The ground matrix for A.
        b_f_ground (np.ndarray): The ground vector for b_f.
        inc_mat (np.ndarray): The incidence matrix of the circuit.
        gnd_idx (int): The index of the ground node.
        b_f (np.ndarray): The combined vector of voltage and current sources.

    Returns:
        tuple: Contains the following:
            i_vec (np.ndarray): The current vector obtained from x (in mL/min).
            p_vec (np.ndarray): The pressure vector obtained from x.
            b_f (np.ndarray): The updated b_f vector.
    """
    x_ground = np.linalg.solve(A_ground, b_f_ground)
    x = ground_calculation(x_ground, inc_mat, gnd_idx)
    i_vec = x[: inc_mat.shape[0]] * 10**6 * 60  # Convert to mL/min
    p_vec = x[inc_mat.shape[0] :]

    if np.isnan(b_f).any():
        b_f = ground_off(b_f_ground, inc_mat, x, gnd_idx)

    return i_vec, p_vec, b_f


def solve_pressure_driven(
    A_ground, b_f_ground, inc_mat, ini_list_inlet, ini_list_outlet, gnd_idx, b_f
):
    """
    Solves the pressure-driven case with known source.

    Args:
        A_ground (np.ndarray): The ground matrix for A.
        b_f_ground (np.ndarray): The ground vector for b_f.
        inc_mat (np.ndarray): The incidence matrix of the circuit.
        ini_list_inlet (list): The list of initial values for the inlets.
        ini_list_outlet (list): The list of initial values for the outlets.
        gnd_idx (int): The index of the ground node.
        b_f (np.ndarray): The combined vector of voltage and current sources.

    Returns:
        tuple: Contains the following:
            i_vec (np.ndarray): The current vector obtained from x (in mL/min).
            p_vec (np.ndarray): The pressure vector obtained from x.
            b_f (np.ndarray): The updated b_f vector.
    """
    # Modify A_ground to handle pressure-driven case
    A_modified = A_ground[:, : -len(ini_list_inlet)]
    mat = np.vstack(
        (
            np.zeros((inc_mat.shape[0], len(ini_list_inlet))),
            np.zeros((len(ini_list_outlet) - 1, len(ini_list_inlet))),
            np.zeros(
                (
                    inc_mat.shape[1] - len(ini_list_inlet) - len(ini_list_outlet),
                    len(ini_list_inlet),
                )
            ),
            -1 * np.eye(len(ini_list_inlet)),
        )
    )
    A_modified = np.hstack((A_modified, mat))

    # Modify b_f_ground
    b_f_modified = b_f_ground[: -len(ini_list_inlet)].copy()
    p_inlet = b_f_ground[-len(ini_list_inlet) :]
    b_f_modified = np.hstack((b_f_modified, np.zeros(len(ini_list_inlet))))

    # Adjust for inlet pressures
    vec = np.hstack(
        (
            inc_mat[:, -len(ini_list_inlet) :] @ p_inlet,
            np.zeros(len(ini_list_outlet) - 1),
            np.zeros(inc_mat.shape[1] - len(ini_list_inlet) - len(ini_list_outlet)),
            np.zeros(len(ini_list_inlet)),
        )
    )
    b_f_modified -= vec

    # Solve the modified system
    x_ground = np.linalg.solve(A_modified, b_f_modified)

    # Reconstruct the full solution
    x = ground_calculation(x_ground, inc_mat, gnd_idx)

    # Extract current and pressure vectors
    i_vec = x[: inc_mat.shape[0]] * 10**6 * 60  # Convert to ,L/min
    p_vec = x[inc_mat.shape[0] :]

    print("Current vector (mL/min):", i_vec)
    print("Pressure vector (Pa):", p_vec)

    # Update b_f if necessary
    if np.isnan(b_f).any():
        b_f = ground_off(b_f_ground, inc_mat, x, gnd_idx)

    return i_vec, p_vec, b_f


def solve_unknown_source_current_driven(
    A, b_f, inc_mat, ini_list, ini_list_outlet, ini_list_inlet
):
    """
    Solves the current-driven case with unknown source.

    This function handles the scenario where the current source is unknown
    and the circuit is current-driven.

    Args:
        A (np.ndarray): The constructed block matrix to solve the problem.
        b_f (np.ndarray): The combined vector of voltage and current sources.
        inc_mat (np.ndarray): The incidence matrix of the circuit.
        ini_list_outlet (list): The list of initial values for the outlets.
        ini_list_inlet (list): The list of initial values for the inlets.

    Returns:
        tuple: Contains the following:
            i_vec (np.ndarray): The current vector obtained from x (in mL/min).
            p_vec (np.ndarray): The pressure vector obtained from x.
            b_f (np.ndarray): The updated b_f vector.
    """
    # Construct matrix for unknown outlet currents
    mat = np.vstack(
        (
            np.zeros((inc_mat.shape[0], len(ini_list_outlet))),
            -1 * np.eye(len(ini_list_outlet)),
            np.zeros(
                (
                    inc_mat.shape[1] - len(ini_list_outlet) - len(ini_list_inlet),
                    len(ini_list_outlet),
                )
            ),
            np.zeros((len(ini_list_inlet), len(ini_list_outlet))),
        )
    )

    # Modify the incidence matrix
    inc_mat_modified = np.hstack((A, mat))

    print(
        np.linalg.matrix_rank(inc_mat_modified, tol=1e-30)
    )  # 21 * 24 matrix with rank 21 -> add 3 equation to make it full rank (pressure condition?)

    # Modify b_f vector
    b_f_modified = b_f.copy()
    for i in range(b_f.shape[0]):
        if inc_mat.shape[0] <= i < inc_mat.shape[0] + len(ini_list_outlet):
            b_f_modified[i] = 0

    # Add pressure constraint equations
    p_const_mat = np.hstack(
        (
            np.zeros((len(ini_list_outlet), inc_mat.shape[0])),
            np.eye(len(ini_list_outlet)),
            np.zeros((len(ini_list_outlet), inc_mat.shape[1])),
        )
    )
    inc_mat_modified = np.vstack((inc_mat_modified, p_const_mat))
    b_f_modified = np.hstack((b_f_modified, np.zeros(len(ini_list_outlet))))

    print(np.linalg.matrix_rank(inc_mat_modified, tol=1e-30))

    # Solve the modified system
    soln = np.linalg.solve(inc_mat_modified, b_f_modified[..., np.newaxis])[..., 0]

    # Extract solution components
    x = soln[: -len(ini_list_outlet)]
    i_vec = x[: inc_mat.shape[0]] * 10**6 * 60  # Convert to mL/min
    p_vec = x[inc_mat.shape[0] :]
    f_o = soln[inc_mat.shape[0] + inc_mat.shape[1] :]

    # Update b_f with solved outlet flows
    for i, item in enumerate(ini_list):
        if item[0] == "f" and np.isnan(item[2]):
            b_f[inc_mat.shape[0] + item[1]] = f_o[i]
            if i >= len(ini_list_outlet) - 1:
                break

    return i_vec, p_vec, b_f


def solve_unknown_source_pressure_driven(
    A, b_f, inc_mat, ini_list, ini_list_outlet, ini_list_inlet
):
    """
    Solves the pressure-driven case with unknown source.

    This function handles the scenario where the pressure source is unknown
    and the circuit is pressure-driven.

    Args:
        A (np.ndarray): The constructed block matrix to solve the problem.
        b_f (np.ndarray): The combined vector of voltage and current sources.
        inc_mat (np.ndarray): The incidence matrix of the circuit.
        ini_list_outlet (list): The list of initial values for the outlets.
        ini_list_inlet (list): The list of initial values for the inlets.

    Returns:
        tuple: Contains the following:
            i_vec (np.ndarray): The current vector obtained from x (in mL/min).
            p_vec (np.ndarray): The pressure vector obtained from x.
            b_f (np.ndarray): The updated b_f vector.
    """
    # Construct matrix for unknown outlet and inlet flows
    mat = np.vstack(
        (
            np.zeros((inc_mat.shape[0], len(ini_list_outlet) + len(ini_list_inlet))),
            np.hstack(
                (
                    -1 * np.eye(len(ini_list_outlet)),
                    np.zeros((len(ini_list_outlet), len(ini_list_inlet))),
                )
            ),
            np.zeros(
                (
                    inc_mat.shape[1] - len(ini_list),
                    len(ini_list_outlet) + len(ini_list_inlet),
                )
            ),
            np.hstack(
                (
                    np.zeros((len(ini_list_inlet), len(ini_list_outlet))),
                    -1 * np.eye(len(ini_list_inlet)),
                )
            ),
        )
    )

    # Modify the incidence matrix
    inc_mat_modified = np.hstack((A[:, : -len(ini_list_inlet)], mat))
    print(np.linalg.matrix_rank(inc_mat_modified, tol=1e-30))

    # Modify b_f vector
    b_f_modified = b_f.copy()
    p_inlet = -1 * b_f[-len(ini_list_inlet) :]
    for i in range(b_f.shape[0]):
        if inc_mat.shape[0] <= i < inc_mat.shape[0] + len(ini_list_outlet):
            b_f_modified[i] = 0
        if i >= inc_mat.shape[0] + inc_mat.shape[1] - len(ini_list_inlet):
            b_f_modified[i] = 0

    # Adjust for inlet pressures
    vec = np.hstack(
        (
            inc_mat[:, -len(ini_list_inlet) :] @ p_inlet,
            np.zeros(len(ini_list_outlet)),
            np.zeros(inc_mat.shape[1] - len(ini_list)),
            np.zeros(len(ini_list_inlet)),
        )
    )
    b_f_modified -= vec

    # Add pressure constraint equations
    p_const_mat = np.hstack(
        (
            np.zeros((len(ini_list_outlet), inc_mat.shape[0])),
            np.eye(len(ini_list_outlet)),
            np.zeros((len(ini_list_outlet), inc_mat.shape[1])),
        )
    )
    inc_mat_modified = np.vstack((inc_mat_modified, p_const_mat))
    b_f_modified = np.hstack((b_f_modified, np.zeros(len(ini_list_outlet))))

    # Solve the modified system
    soln = np.linalg.solve(inc_mat_modified, b_f_modified[..., np.newaxis])[..., 0]

    # Extract solution components
    x = soln[: -len(ini_list_outlet) - len(ini_list_inlet)]
    x = np.hstack((x, p_inlet))
    i_vec = x[: inc_mat.shape[0]] * 10**6 * 60  # Convert to mL/min
    p_vec = x[inc_mat.shape[0] :]
    f_o = soln[
        inc_mat.shape[0] + inc_mat.shape[1] - len(ini_list_inlet) : -len(ini_list_inlet)
    ]
    f_i = soln[-len(ini_list_inlet) :]

    # Update b_f with solved outlet and inlet flows
    for i, item in enumerate(ini_list):
        if item[0] == "f":
            if np.isnan(item[2]):
                if item[3] == "o":
                    b_f[inc_mat.shape[0] + item[1]] = f_o[i]
                elif item[3] == "i":
                    b_f[
                        inc_mat.shape[0] + inc_mat.shape[1] - len(ini_list_inlet) + i
                    ] = f_i[i]

    return i_vec, p_vec, b_f


def Kirchhoff_solver_vec(
    A_vec,
    A_ground_vec,
    b_f_vec,
    b_f_ground_vec,
    inc_mat_vec,
    isSourceUnknown,
    ini_list,
    ini_list_outlet,
    ini_list_inlet,
    gnd_idx,
    popSize,
):
    """
    Solves the Kirchhoff's circuit equations for multiple microfluidic circuits simultaneously.

    This function handles both current-driven and pressure-driven scenarios,
    as well as cases where the source is known or unknown, for multiple circuits at once.

    Args:
        A_vec (np.ndarray): The constructed block matrices to solve the problems.
        A_ground_vec (np.ndarray): The ground matrices for A_vec.
        b_f_vec (np.ndarray): The combined vectors of voltage and current sources.
        b_f_ground_vec (np.ndarray): The ground vectors for b_f_vec.
        inc_mat_vec (np.ndarray): The incidence matrices of the circuits.
        isSourceUnknown (bool): True if the source is unknown, False otherwise.
        ini_list (list): Initial values of voltage & current sources.
        ini_list_outlet (list): The ini_list only with outlet streams.
        ini_list_inlet (list): The ini_list only with inlet streams.
        gnd_idx (int): The index of the ground node.
        popSize (int): The number of circuits to solve simultaneously.

    Returns:
        tuple: Contains the following:
            i_vec (np.ndarray): The current vectors obtained from x_vec (in mL/min).
            p_vec (np.ndarray): The pressure vectors obtained from x_vec.
            b_f_vec (np.ndarray): The updated b_f_vec vectors.
    """
    # Determine if the system is current-driven or pressure-driven
    mode = next((i[4] for i in ini_list if i[3] == "i"), None)
    mode = "Q_driven" if mode == "m^3/s" else "P_driven"

    if not isSourceUnknown:
        if mode == "Q_driven":
            return solve_current_driven_vec(
                A_ground_vec, b_f_ground_vec, inc_mat_vec, gnd_idx, b_f_vec, popSize
            )

        elif mode == "P_driven":
            return solve_pressure_driven_vec(
                A_ground_vec,
                b_f_ground_vec,
                inc_mat_vec,
                ini_list_inlet,
                ini_list_outlet,
                gnd_idx,
                b_f_vec,
                popSize,
            )

    else:
        if mode == "Q_driven":
            return solve_unknown_source_current_driven_vec(
                A_vec, b_f_vec, inc_mat_vec, ini_list_outlet, ini_list_inlet, popSize
            )

        elif mode == "P_driven":
            return solve_unknown_source_pressure_driven_vec(
                A_vec, b_f_vec, inc_mat_vec, ini_list_outlet, ini_list_inlet, popSize
            )


def solve_current_driven_vec(
    A_ground_vec, b_f_ground_vec, inc_mat_vec, gnd_idx, b_f_vec, popSize
):
    """
    Solves the current-driven case with known source for multiple circuits.
    Args:
        A_ground_vec (np.ndarray): The ground matrices for A.
        b_f_ground_vec (np.ndarray): The ground vectors for b_f.
        inc_mat_vec (np.ndarray): The incidence matrices of the circuits.
        gnd_idx (int): The index of the ground node.
        b_f_vec (np.ndarray): The combined vectors of voltage and current sources.
        popSize (int): The number of circuits to solve simultaneously.

    Returns:
        tuple: Contains the following:
            i_vec (np.ndarray): The current vectors obtained from x_vec (in mL/min).
            p_vec (np.ndarray): The pressure vectors obtained from x_vec.
            b_f_vec (np.ndarray): The updated b_f_vec vectors.
    """
    x_ground_vec = np.linalg.solve(A_ground_vec, b_f_ground_vec)
    x_vec = ground_calculation_vec(x_ground_vec, inc_mat_vec, gnd_idx)
    i_vec = x_vec[:, : inc_mat_vec.shape[1]] * 10**6 * 60  # Convert to mL/min
    p_vec = x_vec[:, inc_mat_vec.shape[1] :]

    if np.isnan(b_f_vec).any():
        b_f_vec = ground_off_vec(b_f_ground_vec, inc_mat_vec, x_vec, gnd_idx)

    return i_vec, p_vec, b_f_vec


def solve_pressure_driven_vec(
    A_ground_vec,
    b_f_ground_vec,
    inc_mat_vec,
    ini_list_inlet,
    ini_list_outlet,
    gnd_idx,
    b_f_vec,
    popSize,
):
    """
    Solves the pressure-driven case with known source for multiple circuits simultaneously.

    Args:
        A_ground_vec (np.ndarray): The ground matrices for A.
        b_f_ground_vec (np.ndarray): The ground vectors for b_f.
        inc_mat_vec (np.ndarray): The incidence matrices of the circuits.
        ini_list_inlet (list): The list of initial values for the inlets.
        ini_list_outlet (list): The list of initial values for the outlets.
        gnd_idx (int): The index of the ground node.
        b_f_vec (np.ndarray): The combined vectors of voltage and current sources.
        popSize (int): The number of circuits to solve simultaneously.

    Returns:
        tuple: Contains the following:
            i_vec (np.ndarray): The current vectors obtained from x_vec (in mL/min).
            p_vec (np.ndarray): The pressure vectors obtained from x_vec.
            b_f_vec (np.ndarray): The updated b_f_vec vectors.
    """
    # Modify A_ground_vec to handle pressure-driven case
    A_modified = A_ground_vec[:, :, : -len(ini_list_inlet)]
    mat = np.vstack(
        (
            np.zeros((inc_mat_vec.shape[1], len(ini_list_inlet))),
            np.zeros((len(ini_list_outlet) - 1, len(ini_list_inlet))),
            np.zeros(
                (
                    inc_mat_vec.shape[2] - len(ini_list_inlet) - len(ini_list_outlet),
                    len(ini_list_inlet),
                )
            ),
            -1 * np.eye(len(ini_list_inlet)),
        )
    )
    mat_vec = np.broadcast_to(mat, (popSize,) + mat.shape)
    A_modified = np.concatenate((A_modified, mat_vec), axis=2)

    # Modify b_f_ground_vec
    b_f_modified = np.copy(b_f_ground_vec[:, : -len(ini_list_inlet)])
    p_inlet = b_f_ground_vec[:, -len(ini_list_inlet) :]
    b_f_modified = np.concatenate(
        (b_f_modified, np.zeros((popSize, len(ini_list_inlet)))), axis=1
    )

    # Adjust for inlet pressures
    vec = np.hstack(
        (
            np.einsum("ijk,ik->ij", inc_mat_vec[:, :, -len(ini_list_inlet) :], p_inlet),
            np.zeros((popSize, len(ini_list_outlet) - 1)),
            np.zeros(
                (
                    popSize,
                    inc_mat_vec.shape[2] - len(ini_list_inlet) - len(ini_list_outlet),
                )
            ),
            np.zeros((popSize, len(ini_list_inlet))),
        )
    )
    b_f_modified -= vec

    # Solve the modified system
    x_ground_vec = np.linalg.solve(A_modified, b_f_modified)

    # Reconstruct the full solution
    x_vec = ground_calculation_vec(x_ground_vec, inc_mat_vec, gnd_idx)

    # Extract current and pressure vectors
    i_vec = x_vec[:, : inc_mat_vec.shape[1]] * 10**6 * 60  # Convert to mL/min
    p_vec = x_vec[:, inc_mat_vec.shape[1] :]

    print("Current vector (mL/min):", i_vec)
    print("Pressure vector (Pa):", p_vec)

    # Update b_f_vec if necessary
    if np.isnan(b_f_vec).any():
        b_f_vec = ground_off_vec(b_f_ground_vec, inc_mat_vec, x_vec, gnd_idx)

    return i_vec, p_vec, b_f_vec


def solve_unknown_source_current_driven_vec(
    A_vec, b_f_vec, inc_mat_vec, ini_list_outlet, ini_list_inlet, popSize
):
    """
    Solves the current-driven case with unknown source for multiple circuits simultaneously.

    Args:
        A_vec (np.ndarray): The constructed block matrices to solve the problems.
        b_f_vec (np.ndarray): The combined vectors of voltage and current sources.
        inc_mat_vec (np.ndarray): The incidence matrices of the circuits.
        ini_list_outlet (list): The list of initial values for the outlets.
        ini_list_inlet (list): The list of initial values for the inlets.
        popSize (int): The number of circuits to solve simultaneously.

    Returns:
        tuple: Contains the following:
            i_vec (np.ndarray): The current vectors obtained from x_vec (in mL/min).
            p_vec (np.ndarray): The pressure vectors obtained from x_vec.
            b_f_vec (np.ndarray): The updated b_f_vec vectors.
    """
    # Construct matrix for unknown outlet currents
    mat = np.vstack(
        (
            np.zeros((inc_mat_vec.shape[1], len(ini_list_outlet))),
            -1 * np.eye(len(ini_list_outlet)),
            np.zeros(
                (
                    inc_mat_vec.shape[2] - len(ini_list_outlet) - len(ini_list_inlet),
                    len(ini_list_outlet),
                )
            ),
            np.zeros((len(ini_list_inlet), len(ini_list_outlet))),
        )
    )
    mat_vec = np.broadcast_to(mat, (popSize,) + mat.shape)

    # Modify the incidence matrix
    inc_mat_modified = np.concatenate((A_vec, mat_vec), axis=2)
    print(
        np.linalg.matrix_rank(inc_mat_modified[0], tol=1e-30)
    )  # 21 * 24 matrix with rank 21 -> add 3 equation to make it full rank (pressure condition?)

    # Modify b_f vector
    b_f_modified = np.copy(b_f_vec)
    b_f_modified[
        :, inc_mat_vec.shape[1] : inc_mat_vec.shape[1] + len(ini_list_outlet)
    ] = 0

    # Add pressure constraint equations
    p_const_mat = np.hstack(
        (
            np.zeros((len(ini_list_outlet), inc_mat_vec.shape[1])),
            np.eye(len(ini_list_outlet)),
            np.zeros((len(ini_list_outlet), inc_mat_vec.shape[2])),
        )
    )
    p_const_mat_vec = np.broadcast_to(p_const_mat, (popSize,) + p_const_mat.shape)
    inc_mat_modified = np.concatenate((inc_mat_modified, p_const_mat_vec), axis=1)
    b_f_modified = np.concatenate(
        (b_f_modified, np.zeros((popSize, len(ini_list_outlet)))), axis=1
    )

    print(np.linalg.matrix_rank(inc_mat_modified[0], tol=1e-30))

    # Solve the modified system
    soln = np.linalg.solve(inc_mat_modified, b_f_modified[..., np.newaxis])[..., 0]

    # Extract solution components
    x_vec = soln[:, : -len(ini_list_outlet)]
    i_vec = x_vec[:, : inc_mat_vec.shape[1]] * 10**6 * 60  # Convert to mL/min
    p_vec = x_vec[:, inc_mat_vec.shape[1] :]
    f_o = soln[:, inc_mat_vec.shape[1] + inc_mat_vec.shape[2] :]

    # Update b_f_vec with solved outlet flows
    b_f_vec = np.copy(b_f_vec)
    for i, item in enumerate(ini_list_outlet):
        if np.isnan(item[2]):
            b_f_vec[:, inc_mat_vec.shape[1] + item[1]] = f_o[:, i]

    return i_vec, p_vec, b_f_vec


def solve_unknown_source_pressure_driven_vec(
    A_vec, b_f_vec, inc_mat_vec, ini_list_outlet, ini_list_inlet, popSize
):
    """
    Solves the pressure-driven case with unknown source for multiple circuits simultaneously.

    Args:
        A_vec (np.ndarray): The constructed block matrices to solve the problems.
        b_f_vec (np.ndarray): The combined vectors of voltage and current sources.
        inc_mat_vec (np.ndarray): The incidence matrices of the circuits.
        ini_list_outlet (list): The list of initial values for the outlets.
        ini_list_inlet (list): The list of initial values for the inlets.
        popSize (int): The number of circuits to solve simultaneously.

    Returns:
        tuple: Contains the following:
            i_vec (np.ndarray): The current vectors obtained from x_vec (in mL/min).
            p_vec (np.ndarray): The pressure vectors obtained from x_vec.
            b_f_vec (np.ndarray): The updated b_f_vec vectors.
    """
    # Construct matrix for unknown outlet and inlet flows
    mat = np.vstack(
        (
            np.zeros(
                (inc_mat_vec.shape[1], len(ini_list_outlet) + len(ini_list_inlet))
            ),
            np.hstack(
                (
                    -1 * np.eye(len(ini_list_outlet)),
                    np.zeros((len(ini_list_outlet), len(ini_list_inlet))),
                )
            ),
            np.zeros(
                (
                    inc_mat_vec.shape[2] - len(ini_list_outlet) - len(ini_list_inlet),
                    len(ini_list_outlet) + len(ini_list_inlet),
                )
            ),
            np.hstack(
                (
                    np.zeros((len(ini_list_inlet), len(ini_list_outlet))),
                    -1 * np.eye(len(ini_list_inlet)),
                )
            ),
        )
    )
    mat_vec = np.broadcast_to(mat, (popSize,) + mat.shape)

    # Modify the incidence matrix
    inc_mat_modified = np.concatenate(
        (A_vec[:, :, : -len(ini_list_inlet)], mat_vec), axis=2
    )

    # Modify b_f vector
    b_f_modified = np.copy(b_f_vec)
    p_inlet = -1 * b_f_vec[:, -len(ini_list_inlet) :]
    b_f_modified[
        :, inc_mat_vec.shape[1] : inc_mat_vec.shape[1] + len(ini_list_outlet)
    ] = 0
    b_f_modified[:, -len(ini_list_inlet) :] = 0

    # Adjust for inlet pressures
    vec = np.hstack(
        (
            np.einsum("ijk,ik->ij", inc_mat_vec[:, :, -len(ini_list_inlet) :], p_inlet),
            np.zeros((popSize, len(ini_list_outlet))),
            np.zeros(
                (
                    popSize,
                    inc_mat_vec.shape[2] - len(ini_list_outlet) - len(ini_list_inlet),
                )
            ),
            np.zeros((popSize, len(ini_list_inlet))),
        )
    )
    b_f_modified -= vec

    # Add pressure constraint equations
    p_const_mat = np.hstack(
        (
            np.zeros((len(ini_list_outlet), inc_mat_vec.shape[1])),
            np.eye(len(ini_list_outlet)),
            np.zeros((len(ini_list_outlet), inc_mat_vec.shape[2])),
        )
    )
    p_const_mat_vec = np.broadcast_to(p_const_mat, (popSize,) + p_const_mat.shape)
    inc_mat_modified = np.concatenate((inc_mat_modified, p_const_mat_vec), axis=1)
    b_f_modified = np.concatenate(
        (b_f_modified, np.zeros((popSize, len(ini_list_outlet)))), axis=1
    )

    # Solve the modified system
    soln = np.linalg.solve(inc_mat_modified, b_f_modified[..., np.newaxis])[..., 0]

    # Extract solution components
    x_vec = soln[:, : -len(ini_list_outlet) - len(ini_list_inlet)]
    x_vec = np.concatenate((x_vec, p_inlet), axis=1)
    i_vec = x_vec[:, : inc_mat_vec.shape[1]] * 10**6 * 60  # Convert to mL/min
    p_vec = x_vec[:, inc_mat_vec.shape[1] :]
    f_o = soln[
        :,
        inc_mat_vec.shape[1]
        + inc_mat_vec.shape[2]
        - len(ini_list_inlet) : -len(ini_list_inlet),
    ]
    f_i = soln[:, -len(ini_list_inlet) :]

    # Update b_f_vec with solved outlet and inlet flows
    b_f_vec = np.copy(b_f_vec)
    for i, item in enumerate(ini_list_outlet):
        if np.isnan(item[2]):
            b_f_vec[:, inc_mat_vec.shape[1] + item[1]] = f_o[:, i]

    for i, item in enumerate(ini_list_inlet):
        if item[4] == "Pa":
            b_f_vec[
                :, inc_mat_vec.shape[1] + inc_mat_vec.shape[2] - len(ini_list_inlet) + i
            ] = f_i[:, i]

    return i_vec, p_vec, b_f_vec


def ground_calculation(x_ground, inc_mat, gnd_idx):
    """
    Calculate the full solution vector by reinserting the ground node value.

    This function takes the solution vector obtained from solving the grounded system
    and reinserts the ground node value (which is zero) at the appropriate index.

    Args:
        x_ground (np.ndarray): Solution vector from the ground block matrix.
        inc_mat (np.ndarray): Incidence matrix of the circuit.
        gnd_idx (int): Index of the ground node.

    Returns:
        x (np.ndarray): The complete solution vector with the ground node value reinserted.
    """
    # Insert a zero (ground potential) at the index of the ground node
    x = np.insert(x_ground, inc_mat.shape[0] + gnd_idx, 0)
    return x


def ground_calculation_vec(x_ground_vec, inc_mat_vec, gnd_idx):
    """
    Calculate the full solution vectors by reinserting the ground node values.

    This function takes the solution vectors obtained from solving the grounded systems
    and reinserts the ground node values (which are zero) at the appropriate indices.

    Args:
        x_ground_vec (np.ndarray): Solution vectors from the ground block matrices.
            Shape: (popSize, num_variables - 1)
        inc_mat_vec (np.ndarray): Incidence matrices of the circuits.
            Shape: (popSize, num_nodes, num_edges)
        gnd_idx (int): Index of the ground node.

    Returns:
        x_vec (np.ndarray): The complete solution vectors with the ground node values reinserted.
            Shape: (popSize, num_variables)
    """
    # Insert zeros (ground potentials) at the indices of the ground nodes
    x_vec = np.insert(x_ground_vec, inc_mat_vec.shape[1] + gnd_idx, 0, axis=1)
    return x_vec


def ground_off(b_f_ground, inc_mat, x, gnd_idx):
    """
    Reconstruct the full b_f vector by reinserting the ground node value.

    This function takes the grounded b_f vector and the solution vector,
    and reconstructs the full b_f vector by calculating and reinserting
    the value for the ground node.

    Args:
        b_f_ground (np.ndarray): The grounded b_f vector.
        inc_mat (np.ndarray): The incidence matrix of the circuit.
        x (np.ndarray): The solution vector.
        gnd_idx (int): The index of the ground node.

    Returns:
        b_f (np.ndarray): The full b_f vector with the ground node value reinserted.
    """
    # Calculate the value for the ground node
    b_f_unks = inc_mat.T[gnd_idx] @ x[: inc_mat.shape[0]]

    # Reinsert the calculated value into the b_f vector
    b_f = np.insert(b_f_ground, inc_mat.shape[0] + gnd_idx, b_f_unks)

    return b_f


def ground_off_vec(b_f_ground_vec, inc_mat_vec, x_vec, gnd_idx):
    """
    Reconstruct the full b_f vectors by reinserting the ground node values.

    This function takes the grounded b_f vectors and the solution vectors,
    and reconstructs the full b_f vectors by calculating and reinserting
    the values for the ground nodes.

    Args:
        b_f_ground_vec (np.ndarray): The grounded b_f vectors.
            Shape: (popSize, num_variables - 1)
        inc_mat_vec (np.ndarray): The incidence matrices of the circuits.
            Shape: (popSize, num_nodes, num_edges)
        x_vec (np.ndarray): The solution vectors.
            Shape: (popSize, num_variables)
        gnd_idx (int): The index of the ground node.

    Returns:
        b_f_vec (np.ndarray): The full b_f vectors with the ground node values reinserted.
            Shape: (popSize, num_variables)
    """
    # Calculate the values for the ground nodes
    b_f_unks = np.einsum(
        "ijk,ik->ij",
        np.transpose(inc_mat_vec, axes=(0, 2, 1))[:, gnd_idx],
        x_vec[:, : inc_mat_vec.shape[1]],
    )

    # Reinsert the calculated values into the b_f vectors
    b_f_vec = np.insert(
        b_f_ground_vec, inc_mat_vec.shape[1] + gnd_idx, b_f_unks, axis=1
    )

    return b_f_vec


def save_currvol_csv(i_vec, p_vec, nametag, address):
    """
    Save current and voltage vectors into CSV files and return DataFrames.

    This function creates DataFrames for current (flow rate) and voltage (pressure) vectors,
    saves them as CSV files if a nametag is provided.

    Args:
        i_vec (np.ndarray): A solved current vector of each edge (flow rates in mL/min).
        p_vec (np.ndarray): A solved voltage vector of each node (pressures in Pa).
        nametag (str or None): Name tag to distinguish each CSV file for repeated function execution.
        address (str): File path for saving the CSV files.

    Returns:
        None
    """
    # Create DataFrames for current and voltage vectors
    current_df = pd.DataFrame({"flow_rate(ml/min)": i_vec})
    voltage_df = pd.DataFrame({"pressure(Pa)": p_vec})

    # Save CSV files if a nametag is provided
    if not nametag == None:
        current_df.to_csv(f"{address}/current_vec_{nametag}.csv")
        voltage_df.to_csv(f"{address}/voltage_vec_{nametag}.csv")


def plotting_outlet(
    i_vec,
    outlet_edge_idx,
    plot_type,
    address,
    whatToSolve,
    ini_list,
    args,
    conc_vec=None,
    nametag=None,
    inc_csv=None,
    length_csv=None,
    conc_csv=None,
):
    """
    Plot outlet flow rates and concentrations.

    This function creates bar plots for outlet flow rates and concentrations (if available).

    Args:
        i_vec (np.ndarray): Solved current vector of each edge (flow rates in mL/min).
        outlet_edge_idx (list): List of outlet edge indices.
        plot_type (tuple): Designating plot types ('compare' and/or 'concentration').
        address (str): File path for saving the plots.
        whatToSolve (str or list): Indicates what was solved in the problem.
        conc_vec (list of np.ndarray, optional): Concentration vectors for each inlet. Defaults to None.
        nametag (str, optional): Name tag to distinguish each plot for repeated function execution. Defaults to None.

    Returns:
        None
    """
    plt.rcParams.update({
        'font.size': 50,
        'text.usetex': True,
        'font.family': 'times',
        'axes.linewidth': 5,
        'lines.linewidth': 5,
        'xtick.major.width':5,
        'ytick.major.width':5,
        'xtick.major.size':10,
        'ytick.major.size':10,
        'xtick.major.pad':10,
        'ytick.major.pad':10,

        'xtick.minor.width':5,
        'ytick.minor.width':5,
        'xtick.minor.size':10,
        'ytick.minor.size':10,
    
        'legend.facecolor':'white',
        'legend.edgecolor':'None',
        'legend.framealpha':0.5
    })

    tag = "revised" if "length" in whatToSolve else "original"

    outlets = outlet_combine(i_vec, outlet_edge_idx)

    if "compare" in plot_type:
        plt.figure(figsize=(15, 12))
        plt.bar(
            ["O" + str(i + 1) for i in range(len(outlet_edge_idx))],
            [abs(outlets[i]) for i in range(len(outlet_edge_idx))],
        )
        plt.title("Outlet Flow Rate")
        plt.xlabel("Outlet")
        plt.ylabel("Flow Rate (ml/min)")
        plt.tight_layout()
        plt.savefig(f"{address}/outlet_compare_{tag}_{nametag}.png")
        plt.close()

    if "concentration" in plot_type and conc_vec is not None:
        plt.figure(figsize=(15, 12))
        x = np.arange(len(outlet_edge_idx))
        width = 0.8 / len(conc_vec)
        for idx, conc in enumerate(conc_vec):
            plt.bar(
                x + width * (idx - len(conc_vec) / 2 + 0.5),
                conc,
                width=width,
                label=f"inlet_{idx+1}",
            )
        plt.title("Outlet Concentration")
        plt.xlabel("Outlet")
        plt.ylabel("Concentration")
        plt.xticks(x, [f"O{j+1}" for j in range(len(outlet_edge_idx))])
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{address}/outlet_concentration_{tag}_{nametag}.png")
        plt.close()
        
    if "difference" in plot_type:
        # Load and prepare initial data
        length_df = pd.read_csv(f"{address}/{length_csv}", index_col="edge")
        length = length_df["length"].to_numpy()
        
        # Prepare concentration data
        tmp_conc_df = conc_mat_produce(conc_csv, address)
        target_conc_df = []
        tmp_conc_df_col = list(tmp_conc_df.columns)
        for i in range(len(tmp_conc_df.columns)):
            tmp_col = copy.deepcopy(tmp_conc_df_col)
            tmp_col.remove(f"fraction{i+1}")
            tmp = tmp_conc_df.drop(columns=tmp_col)
            target_conc_df.append(tmp)

        target_conc_vec = [i.to_numpy() for i in target_conc_df]
        target_conc_vec_total = np.vstack(tuple(target_conc_vec)).T.flatten()
        target_conc_vec = [i.T.flatten() for i in target_conc_vec]

        # Determine flow mode and calculate inlet/outlet currents
        (
            inlet_current,
            outlet_current,
            outlet_current_vec,
            outlet_current_vec_scaled,
        ) = calculate_currents(whatToSolve, inc_csv, length, ini_list, args, address)

        # Set up the optimization problem
        constraint = [outlet_current_vec, target_conc_vec_total]
        if "compare" in plot_type:
            plt.figure(figsize=(15, 12))
            plt.bar(
                ["O" + str(i + 1) for i in range(len(outlet_edge_idx))],
                [abs(abs(outlets[i]) - abs(outlet_current_vec[i])) for i in range(len(outlet_edge_idx))],
            )
            plt.title("Flow Rate Difference")
            plt.xlabel("Outlet")
            plt.ylabel("Flow Rate (ml/min)")
            plt.tight_layout()
            plt.savefig(f"{address}/outlet_flow_rate_difference_{tag}_{nametag}.png")
            plt.close()
            
        if "concentration" in plot_type and conc_vec is not None:
            plt.figure(figsize=(15, 12))
            x = np.arange(len(outlet_edge_idx))
            width = 0.8 / len(conc_vec)
            for idx, conc in enumerate(conc_vec):
                plt.bar(
                    x + width * (idx - len(conc_vec) / 2 + 0.5),
                    abs(conc - target_conc_vec[idx]),
                    width=width,
                    label=f"inlet_{idx+1}",
                )
            plt.title("Concentration Difference")
            plt.xlabel("Outlet")
            plt.ylabel("Concentration")
            plt.xticks(x, [f"O{j+1}" for j in range(len(outlet_edge_idx))])
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{address}/outlet_concentration_difference_{tag}_{nametag}.png")
            plt.close()


def outlet_combine(i_vec, outlet_edge_idx):
    """
    Combine outlet flow rates into a single list.

    This function extracts the flow rates of the outlet edges from the
    current vector and combines them into a single list.

    Args:
        i_vec (np.ndarray): A solved current vector of each edge (flow rates).
        outlet_edge_idx (list): A list of indices designating outlet edges.

    Returns:
        outlets (list): The list of outlet flow rates.
    """
    outlets = []
    for idx in outlet_edge_idx:
        outlets.append(i_vec[idx])
    return outlets


def execute_functions(
    whatToSolve,
    inc_csv,
    length_csv,
    args,
    ini_list,
    inlet_node_idx,
    inlet_edge_idx,
    nametag,
    address,
):
    """
    Execute the main functions to solve the microfluidic circuit problem.

    This function performs the following steps:
    1. Sets up variables based on what needs to be solved.
    2. Produces incidence and length matrices from CSV files.
    3. Constructs the block matrix for the circuit.
    4. Constructs the right-hand side (RHS) vector.
    5. Applies ground node calculations.
    6. Solves the Kirchhoff's equations.
    7. Calculates concentrations if required.
    8. Saves the results to CSV files.

    Args:
        whatToSolve (str or list): Indicates what to solve in the problem.
        inc_csv (str): Filename of the incidence matrix CSV.
        length_csv (str): Filename of the length matrix CSV.
        args (tuple): Contains (eta, w, h, n) where:
            eta (float): Fluid viscosity.
            w (float): Channel width.
            h (float): Channel height.
            n (int): Computation coefficient.
        ini_list (list): Initial values of voltage & current sources.
        inlet_node_idx (list): Indices of inlet nodes.
        inlet_edge_idx (list): Indices of inlet edges.
        nametag (str): Tag for naming output files.
        address (str): Directory path for saving output files.

    Returns:
        tuple: Contains the following:
            i_vec (np.array): Solved current vector for each edge.
            conc_vec (list or None): Concentration vectors if calculated, else None.
    """
    eta, w, h, n = args
    (
        isLengthUnknown,
        isSourceUnknown,
        isConcUnknown,
    ) = variable_setup(whatToSolve)

    # Produce matrices from CSV files
    inc_mat = inc_mat_produce(inc_csv, address)
    length_df = length_mat_produce(length_csv, address)
    length = list(length_df["length"])

    # Prepare lists for inlet and outletini_list_outlet = []
    ini_list_outlet = [i for i in ini_list if i[3] == "o"]
    ini_list_inlet = [i for i in ini_list if i[3] == "i"]

    # Construct block matrix
    arg = (w, h, n)
    A = construct_block_mat(length, eta, arg, inc_mat)

    # Construct RHS vector
    b_f = construct_RHS(inc_mat, ini_list)

    # Apply ground node calculations
    A_ground, b_f_ground, gnd_idx = ground_block_mat(A, inc_mat, b_f, ini_list_outlet)

    # Solve Kirchhoff's equations
    i_vec, p_vec, b_f = Kirchhoff_solver(
        A,
        A_ground,
        b_f,
        b_f_ground,
        inc_mat,
        isSourceUnknown,
        ini_list,
        ini_list_outlet,
        ini_list_inlet,
        gnd_idx,
    )

    # Save results to CSV
    save_currvol_csv(i_vec, p_vec, nametag, address)

    # Calculate concentrations if required
    conc_vec = None
    if isConcUnknown == True and len(ini_list_inlet) > 1:
        conc_vec = conc_calculation(
            b_f,
            i_vec,
            inc_mat,
            ini_list_inlet,
            ini_list_outlet,
            inlet_node_idx,
            inlet_edge_idx,
        )
        conc_df = [pd.DataFrame({"mole fraction": i}) for i in conc_vec]
        for idx, i in enumerate(conc_df):
            i.to_csv(f"{address}/conc_vec_inlet_{idx+1}.csv")

    return i_vec, conc_vec


def conc_calculation(
    b_f,
    i_vec,
    inc_mat,
    ini_list_inlet,
    ini_list_outlet,
    inlet_node_idx,
    inlet_edge_idx,
):
    """
    Calculate the concentration vector for a microfluidic circuit.

    This function computes the concentration of different species at each outlet
    based on the circuit parameters and flow rates.

    Args:
        b_f (np.ndarray): The combined vector of voltage and current sources.
        i_vec (np.ndarray): A solved current vector of each edge.
        inc_mat (np.ndarray): Incidence matrix of the circuit.
        ini_list_inlet (list): The ini_list only with inlet streams.
        ini_list_outlet (list): The ini_list only with outlet streams.
        inlet_node_idx (list): Indices of inlet nodes.
        inlet_edge_idx (list): Indices of inlet edges.

    Returns:
        conc_vec (list): The list of concentration vectors for each species at the outlets.
    """
    f = b_f[inc_mat.shape[0] :]

    conc_vec = []

    for inlet in ini_list_inlet:
        # Create inlet concentration vector
        inlet_conc = np.zeros(len(ini_list_inlet))
        inlet_conc[inlet_node_idx.index(inlet[1])] = 1

        # Create flow rate matrix
        i_mat = np.diag(i_vec * 10 ** (-6) / 60)  # Convert to m^3/s

        f_mat = np.diag(f)

        # Divide matrices
        inc_mat_divide = mat_divide(inc_mat.T, ini_list_inlet, ini_list_outlet, dim=2)
        i_mat_divide = mat_divide(i_mat, ini_list_inlet, ini_list_outlet, dim=1)
        f_mat_divide = mat_divide(f_mat, ini_list_inlet, ini_list_outlet, dim=1)

        # Handle nodes with multiple outlets
        two_or_more_outlet_list = []
        for idx in range(f.shape[0]):
            if list(inc_mat.T[idx]).count(-1) > 1:
                two_or_more_outlet_list.append(idx)

        elem_list = []
        for idx in two_or_more_outlet_list:
            pvt = np.zeros(i_vec.shape[0])
            for j in range(i_vec.shape[0]):
                if inc_mat.T[idx][j] == 1:
                    pvt[j] = 1
            for j in range(i_vec.shape[0]):
                if inc_mat.T[idx][j] == -1:
                    tmp = pvt.copy()
                    tmp[j] = -1
                    elem_list.append(tmp)

        two_or_more_outlet_mat = np.vstack(tuple(elem_list)) if elem_list else None
        if two_or_more_outlet_mat is not None:
            two_or_more_outlet_mats = np.hsplit(
                two_or_more_outlet_mat, [inlet_edge_idx[0]]
            )

        # Construct LHS matrix
        LHS = inc_mat_divide[0][0] @ i_mat_divide[0] - f_mat_divide[0]
        LHS = np.hstack((LHS, inc_mat_divide[0][1] @ i_mat_divide[1]))
        LHS = np.vstack(
            (
                LHS,
                np.hstack(
                    (
                        inc_mat_divide[1][0] @ i_mat_divide[0],
                        inc_mat_divide[1][1] @ i_mat_divide[1],
                    )
                ),
            )
        )
        LHS = np.vstack(
            (
                LHS,
                np.hstack(
                    (
                        inc_mat_divide[2][0] @ i_mat_divide[0],
                        inc_mat_divide[2][1] @ i_mat_divide[1],
                    )
                ),
            )
        )

        LHS *= 10 ** (10)  # Scale for numerical stability

        if two_or_more_outlet_mat is not None:
            LHS = np.vstack((LHS, two_or_more_outlet_mats[0]))

        # Construct RHS vector
        RHS = np.hstack(
            (
                -1 * inc_mat_divide[0][2] @ i_mat_divide[2] @ inlet_conc,
                -1 * inc_mat_divide[1][2] @ i_mat_divide[2] @ inlet_conc,
                (f_mat_divide[2] - inc_mat_divide[2][2] @ i_mat_divide[2]) @ inlet_conc,
            )
        )

        RHS *= 10 ** (10)  # Scale for numerical stability

        if two_or_more_outlet_mat is not None:
            RHS = np.hstack((RHS, -1 * two_or_more_outlet_mats[1] @ inlet_conc))

        # Solve the system
        LHS_modified = LHS.T @ LHS
        RHS_modified = LHS.T @ RHS

        soln = np.linalg.solve(LHS_modified, RHS_modified[..., np.newaxis])[..., 0]

        conc_vec.append(soln[: len(ini_list_outlet)])

    return conc_vec


def conc_calculation_vec(
    b_f_vec,
    i_vec,
    inc_mat_vec,
    ini_list_inlet,
    ini_list_outlet,
    inlet_node_idx,
    inlet_edge_idx,
    popSize,
):
    """
    Calculate the concentration vectors for multiple microfluidic circuits simultaneously.

    This function computes the concentration of different species at each outlet
    based on the circuit parameters and flow rates for multiple circuits at once.

    Args:
        b_f_vec (np.ndarray): The combined vectors of voltage and current sources.
        i_vec (np.ndarray): Solved current vectors of each edge.
        inc_mat_vec (np.ndarray): Incidence matrices of the circuits.
        ini_list_inlet (list): The ini_list only with inlet streams.
        ini_list_outlet (list): The ini_list only with outlet streams.
        inlet_node_idx (list): Indices of inlet nodes.
        inlet_edge_idx (list): Indices of inlet edges.
        popSize (int): Number of circuits to process in parallel.

    Returns:
        conc_vec (list): The list of concentration vectors for each species at the outlets for all circuits.
    """
    f_vec = b_f_vec[:, inc_mat_vec.shape[1] :]

    conc_vec = []

    for inlet in ini_list_inlet:
        # Create inlet concentration vector
        inlet_conc = np.zeros(len(ini_list_inlet))
        for idx, j in enumerate(inlet_node_idx):
            if inlet[1] == j:
                inlet_conc[idx] = 1

        inlet_conc_vec = np.broadcast_to(inlet_conc, (popSize,) + inlet_conc.shape)

        # Create flow rate matrix
        i_mat = (
            np.where(np.eye(inc_mat_vec.shape[1], dtype=bool), i_vec[:, np.newaxis], 0)
            * 10 ** (-6)
            / 60
        )  # Convert to m^3/s

        # Create source matrix
        f_mat = np.where(
            np.eye(inc_mat_vec.shape[2], dtype=bool), f_vec[:, np.newaxis], 0
        )

        # Divide matrices
        inc_mat_divide = mat_divide_vec(
            np.transpose(inc_mat_vec, axes=(0, 2, 1)),
            ini_list_inlet,
            ini_list_outlet,
            dim=2,
        )
        i_mat_divide = mat_divide_vec(i_mat, ini_list_inlet, ini_list_outlet, dim=1)
        f_mat_divide = mat_divide_vec(f_mat, ini_list_inlet, ini_list_outlet, dim=1)

        # Handle nodes with multiple outlets
        two_or_more_outlet_list = []
        for idx in range(f_vec.shape[1]):
            transposed = np.transpose(inc_mat_vec, axes=(0, 2, 1))[:, idx]
            count_minus_one = np.sum(transposed == -1, axis=1)
            if np.any(count_minus_one > 1):
                two_or_more_outlet_list.append(idx)

        elem_list = []
        for idx in two_or_more_outlet_list:
            pvt = np.zeros(i_vec.shape[1])
            for j in range(i_vec.shape[1]):
                if np.any(
                    np.transpose(inc_mat_vec, axes=(0, 2, 1))[:, idx, j]
                    == np.ones(popSize)
                ):
                    pvt[j] = 1
            for j in range(i_vec.shape[1]):
                if np.any(
                    np.transpose(inc_mat_vec, axes=(0, 2, 1))[:, idx, j]
                    == -1 * np.ones(popSize)
                ):
                    tmp = copy.deepcopy(pvt)
                    tmp[j] = -1
                    elem_list.append(tmp)

        two_or_more_outlet_mat = np.vstack(tuple(elem_list)) if elem_list else None
        two_or_more_outlet_mat = np.broadcast_to(
            two_or_more_outlet_mat, (popSize,) + two_or_more_outlet_mat.shape
        )
        if two_or_more_outlet_mat is not None:
            two_or_more_outlet_mats = np.split(
                two_or_more_outlet_mat, [inlet_edge_idx[0]], axis=2
            )

        # Construct LHS matrix
        LHS = (
            np.einsum("ijk,ikl->ijl", inc_mat_divide[0][0], i_mat_divide[0])
            - f_mat_divide[0]
        )
        LHS = np.concatenate(
            (LHS, np.einsum("ijk,ikl->ijl", inc_mat_divide[0][1], i_mat_divide[1])),
            axis=2,
        )
        LHS = np.concatenate(
            (
                LHS,
                np.concatenate(
                    (
                        np.einsum(
                            "ijk,ikl->ijl", inc_mat_divide[1][0], i_mat_divide[0]
                        ),
                        np.einsum(
                            "ijk,ikl->ijl", inc_mat_divide[1][1], i_mat_divide[1]
                        ),
                    ),
                    axis=2,
                ),
            ),
            axis=1,
        )
        LHS = np.concatenate(
            (
                LHS,
                np.concatenate(
                    (
                        np.einsum(
                            "ijk,ikl->ijl", inc_mat_divide[2][0], i_mat_divide[0]
                        ),
                        np.einsum(
                            "ijk,ikl->ijl", inc_mat_divide[2][1], i_mat_divide[1]
                        ),
                    ),
                    axis=2,
                ),
            ),
            axis=1,
        )

        LHS *= 10 ** (10)  # Scale for numerical stability

        if two_or_more_outlet_mat is not None:
            LHS = np.concatenate((LHS, two_or_more_outlet_mats[0]), axis=1)

        # Construct RHS vector
        RHS = np.concatenate(
            (
                -1
                * np.einsum(
                    "ijk,ik->ij",
                    np.einsum("ijk,ikl->ijl", inc_mat_divide[0][2], i_mat_divide[2]),
                    inlet_conc_vec,
                ),
                -1
                * np.einsum(
                    "ijk,ik->ij",
                    np.einsum("ijk,ikl->ijl", inc_mat_divide[1][2], i_mat_divide[2]),
                    inlet_conc_vec,
                ),
                np.einsum(
                    "ijk,ik->ij",
                    (
                        f_mat_divide[2]
                        - np.einsum(
                            "ijk,ikl->ijl", inc_mat_divide[2][2], i_mat_divide[2]
                        )
                    ),
                    inlet_conc_vec,
                ),
            ),
            axis=1,
        )

        RHS *= 10 ** (10)  # Scale for numerical stability

        if two_or_more_outlet_mat is not None:
            RHS = np.hstack(
                (
                    RHS,
                    -1
                    * np.einsum(
                        "ijk,ik->ij", two_or_more_outlet_mats[1], inlet_conc_vec
                    ),
                )
            )

        # Solve the system
        LHS_modified = np.einsum("ijk,ikl->ijl", np.transpose(LHS, axes=(0, 2, 1)), LHS)
        RHS_modified = np.einsum("ijk,ik->ij", np.transpose(LHS, axes=(0, 2, 1)), RHS)

        soln = np.linalg.solve(LHS_modified, RHS_modified[..., np.newaxis])[..., 0]

        conc_vec.append(soln[:, : len(ini_list_outlet)])

    return conc_vec


def mat_divide(mat, ini_list_inlet, ini_list_outlet, dim):
    """
    Divide the matrix into three parts based on inlet and outlet information.

    This function divides the input matrix into submatrices corresponding to
    outlet, intermediate, and inlet sections of the microfluidic circuit.

    Args:
        mat (np.ndarray): Matrix to divide.
        ini_list_inlet (list): List of initial values for inlet streams.
        ini_list_outlet (list): List of initial values for outlet streams.
        dim (int): Dimension along which to divide the matrix (1 or 2).

    Returns:
        list: List of divided matrices. The structure depends on the dimension:
            If dim == 1:
                [outlet_outlet, intermediate_intermediate, inlet_inlet]
            If dim == 2:
                [[outlet_outlet, outlet_intermediate, outlet_inlet],
                 [intermediate_outlet, intermediate_intermediate, intermediate_inlet],
                 [inlet_outlet, inlet_intermediate, inlet_inlet]]
    """
    n_outlet = len(ini_list_outlet)
    n_inlet = len(ini_list_inlet)

    if dim == 1:
        n_middle = mat.shape[0] - n_outlet - n_inlet

        divide = [
            mat[:n_outlet, :n_outlet],
            mat[
                n_outlet : n_outlet + n_middle,
                n_outlet : n_outlet + n_middle,
            ],
            mat[
                n_outlet + n_middle :,
                n_outlet + n_middle :,
            ],
        ]

        return divide

    elif dim == 2:
        n_middle_row = mat.shape[0] - n_outlet - n_inlet
        n_middle_col = mat.shape[1] - n_outlet - n_inlet

        divide = [
            [
                mat[:n_outlet, :n_outlet],
                mat[
                    :n_outlet,
                    n_outlet : n_outlet + n_middle_col,
                ],
                mat[:n_outlet, n_outlet + n_middle_col :],
            ],
            [
                mat[
                    n_outlet : n_outlet + n_middle_row,
                    :n_outlet,
                ],
                mat[
                    n_outlet : n_outlet + n_middle_row,
                    n_outlet : n_outlet + n_middle_col,
                ],
                mat[
                    n_outlet : n_outlet + n_middle_row,
                    n_outlet + n_middle_col :,
                ],
            ],
            [
                mat[n_outlet + n_middle_row :, :n_outlet],
                mat[
                    n_outlet + n_middle_row :,
                    n_outlet : n_outlet + n_middle_col,
                ],
                mat[
                    n_outlet + n_middle_row :,
                    n_outlet + n_middle_col :,
                ],
            ],
        ]

        return divide

    else:
        raise ValueError("Dimension (dim) must be either 1 or 2")


def mat_divide_vec(mat, ini_list_inlet, ini_list_outlet, dim):
    """
    Divide the matrix into three parts for vectorized operations.

    This function divides the input matrix into submatrices corresponding to
    outlet, intermediate, and inlet sections of multiple microfluidic circuits.

    Args:
        mat (np.ndarray): Matrix to divide. Shape: (popSize, num_rows, num_cols)
        ini_list_inlet (list): List of initial values for inlet streams.
        ini_list_outlet (list): List of initial values for outlet streams.
        dim (int): Dimension along which to divide the matrix (1 or 2).

    Returns:
        list: List of divided matrices. The structure depends on the dimension:
            If dim == 1:
                [outlet_outlet, intermediate_intermediate, inlet_inlet]
            If dim == 2:
                [[outlet_outlet, outlet_intermediate, outlet_inlet],
                 [intermediate_outlet, intermediate_intermediate, intermediate_inlet],
                 [inlet_outlet, inlet_intermediate, inlet_inlet]]

    Raises:
        ValueError: If an invalid dimension is provided.
    """
    n_outlet = len(ini_list_outlet)
    n_inlet = len(ini_list_inlet)

    if dim == 1:
        n_middle = mat.shape[1] - n_outlet - n_inlet

        divide = [
            mat[:, :n_outlet, :n_outlet],
            mat[
                :,
                n_outlet : n_outlet + n_middle,
                n_outlet : n_outlet + n_middle,
            ],
            mat[
                :,
                n_outlet + n_middle :,
                n_outlet + n_middle :,
            ],
        ]

        return divide

    elif dim == 2:
        n_middle_row = mat.shape[1] - n_outlet - n_inlet
        n_middle_col = mat.shape[2] - n_outlet - n_inlet

        divide = [
            [
                mat[:, :n_outlet, :n_outlet],
                mat[
                    :,
                    :n_outlet,
                    n_outlet : n_outlet + n_middle_col,
                ],
                mat[:, :n_outlet, n_outlet + n_middle_col :],
            ],
            [
                mat[
                    :,
                    n_outlet : n_outlet + n_middle_row,
                    :n_outlet,
                ],
                mat[
                    :,
                    n_outlet : n_outlet + n_middle_row,
                    n_outlet : n_outlet + n_middle_col,
                ],
                mat[
                    :,
                    n_outlet : n_outlet + n_middle_row,
                    n_outlet + n_middle_col :,
                ],
            ],
            [
                mat[:, n_outlet + n_middle_row :, :n_outlet],
                mat[
                    :,
                    n_outlet + n_middle_row :,
                    n_outlet : n_outlet + n_middle_col,
                ],
                mat[
                    :,
                    n_outlet + n_middle_row :,
                    n_outlet + n_middle_col :,
                ],
            ],
        ]

        return divide

    else:
        raise ValueError("Dimension (dim) must be either 1 or 2")


def _execute_length_change_nsga2(
    whatToSolve,
    changing_edges,
    inc_csv,
    length_csv,
    conc_csv,
    args,
    ini_list,
    inlet_node_idx,
    inlet_edge_idx,
    outlet_node_idx,
    outlet_edge_idx,
    conc_wght,
    flow_wght,
    address,
    popSize=200,
    nGen=200,
):
    """
    Execute the length change optimization for the microfluidic circuit.

    This function performs optimization to find the best channel lengths
    that satisfy the given constraints and objectives.

    Args:
        whatToSolve (str or list): Indicates what to solve in the problem.
        changing_edges (list): List of edges that can be changed.
        inc_csv (str): Filename of the incidence matrix CSV.
        length_csv (str): Filename of the length matrix CSV.
        conc_csv (str): Filename of the concentration matrix CSV.
        args (tuple): Contains (eta, w, h, n) where:
            eta (float): Fluid viscosity.
            w (float): Channel width.
            h (float): Channel height.
            n (int): Computation coefficient.
        ini_list (list): Initial values of voltage & current sources.
        inlet_node_idx (list): Indices of inlet nodes.
        inlet_edge_idx (list): Indices of inlet edges.
        outlet_node_idx (list): Indices of outlet nodes.
        outlet_edge_idx (list): Indices of outlet edges.
        conc_wght (float): Weight for concentration in the objective function.
        flow_wght (float): Weight for flow in the objective function.
        address (str): Directory path for saving output files.
        popSize (int, optional): Population size for optimization. Defaults to 100.
        nGen (int, optional): Number of generations for optimization. Defaults to 200.

    Returns:
        tuple: Contains the following:
            time_diff (float): Time taken for optimization.
            length_df (pd.DataFrame): DataFrame of the optimized lengths.
    """
    time_begin = time.time()

    # Load and prepare initial data
    length_df = pd.read_csv(f"{address}/{length_csv}", index_col="edge")
    # pandas 3 can expose a read-only NumPy view; the selected design is
    # assigned into this array below, so request an owned writable copy.
    length = length_df["length"].to_numpy(copy=True)

    lb = np.array([i / 3 for i in length])
    ub = np.array([i * 3 for i in length])
    x0 = length

    # Prepare concentration data
    tmp_conc_df = conc_mat_produce(conc_csv, address)
    target_conc_df = []
    tmp_conc_df_col = list(tmp_conc_df.columns)
    for i in range(len(tmp_conc_df.columns)):
        tmp_col = copy.deepcopy(tmp_conc_df_col)
        tmp_col.remove(f"fraction{i+1}")
        tmp = tmp_conc_df.drop(columns=tmp_col)
        target_conc_df.append(tmp)

    target_conc_vec = [i.to_numpy() for i in target_conc_df]
    target_conc_vec_total = np.vstack(tuple(target_conc_vec)).T.flatten()

    # Determine flow mode and calculate inlet/outlet currents
    (
        inlet_current,
        outlet_current,
        outlet_current_vec,
        outlet_current_vec_scaled,
    ) = calculate_currents(whatToSolve, inc_csv, length, ini_list, args, address)

    # Set up the optimization problem
    constraint = [outlet_current_vec_scaled, target_conc_vec_total]

    problem = MicrofluidicOptimizationProblem(
        constraint,
        whatToSolve,
        changing_edges,
        inc_csv,
        length_csv,
        args,
        ini_list,
        inlet_node_idx,
        inlet_edge_idx,
        outlet_node_idx,
        outlet_edge_idx,
        address,
        lb,
        ub,
        flow_wght,
        conc_wght,
        popSize,
    )

    # Create the callback instance and run the optimization
    callback = ConvergenceCallback()
    algorithm = NSGA2(pop_size=popSize)
    res = minimize(problem, algorithm, ("n_gen", nGen), verbose=True, callback=callback)

    time_end = time.time()
    time_diff = time_end - time_begin
    
    # Process and visualize results
    if not "optimization" in whatToSolve:
        process_optimization_results(res, callback, x0, address)

    # min_values = np.min(res.F, axis=0)
    # max_values = np.max(res.F, axis=0)

    # normalized_objectives = (res.F - min_values) / (max_values - min_values)
    # distances = cdist(normalized_objectives, np.array([[0, 0]]))
    # knee_idx = np.argmin(distances)

    # Select a solution from the Pareto-optimal set based on your preference
    all_solutions = res.X
    lengths_df = pd.DataFrame(all_solutions, columns=length_df.index)
    lengths_df.to_csv(f"{address}/new_lengths_mat.csv")
    
    selected_solution = select_optimal_solution(res)
    length[changing_edges] = selected_solution
    length_df["length"] = length

    if not "optimization" in whatToSolve:
        new_length_csv = f"new_length_mat.csv"
        length_df.to_csv(f"{address}/{new_length_csv}")
        print("time: ", time_diff)

    if "optimization" in whatToSolve:
        print("time: ", time_diff)
        return time_diff, length_df

    # changing_edges -> whole edges, lb & ub -> 0 & infty (in this case, 10 can be reasonable)


def calculate_currents(whatToSolve, inc_csv, length, ini_list, args, address):
    """
    Calculate inlet and outlet currents based on initial conditions.

    This function determines whether the system is current-driven or pressure-driven,
    and calculates the total inlet current and outlet currents accordingly.

    Args:
        ini_list (list): List of initial values for voltage and current sources.
            Each item is a list [type, index, value, location, unit].

    Returns:
        tuple: Contains the following:
            inlet_current (float): Total inlet current (flow rate).
            outlet_current (float): Current (flow rate) for each outlet.
            outlet_current_vec (np.ndarray): Vector of outlet currents.
            outlet_current_vec_scaled (np.ndarray): Vector of scaled outlet currents.

    Note:
        For current-driven systems, the inlet current is converted to mL/min.
        For pressure-driven systems, the inlet current is calculated using
        the calculate_pressure_driven_current function.
    """
    inlet_current = sum(i[2] for i in ini_list if i[3] == "i" and i[4] == "m^3/s")
    current_driven = any(i[4] == "m^3/s" for i in ini_list if i[3] == "i")

    if current_driven:
        inlet_current *= 10**6 * 60  # Convert to mL/min
    else:
        inlet_current = calculate_pressure_driven_current(
            whatToSolve, inc_csv, length, ini_list, args, address
        )

    outlet_count = sum(1 for i in ini_list if i[3] == "o")
    outlet_current = inlet_current / outlet_count
    outlet_current_vec = np.array([outlet_current] * outlet_count)
    outlet_current_vec_scaled = outlet_current_vec / inlet_current

    return inlet_current, outlet_current, outlet_current_vec, outlet_current_vec_scaled


def calculate_pressure_driven_current(
    whatToSolve, inc_csv, length, ini_list, args, address
):
    """
    Calculate the total current (flow rate) for pressure-driven flow in a microfluidic system.

    This function solves the Kirchhoff's equations for the microfluidic circuit and computes
    the total inlet current based on the pressure-driven flow.

    Args:
        whatToSolve (str or list): Indicates what to solve in the problem.
        inc_csv (str): Filename of the incidence matrix CSV.
        length (list): List of channel lengths.
        ini_list (list): Initial values of voltage & current sources.
        args (tuple): Contains (eta, w, h, n) where:
            eta (float): Fluid viscosity.
            w (float): Channel width.
            h (float): Channel height.
            n (int): Computation coefficient.
        address (str): File path for reading/saving data.

    Returns:
        inlet_current (float): The calculated total inlet current (flow rate) in the system.

    Note:
        This function assumes pressure-driven flow and calculates the resulting
        current (flow rate) based on the circuit solution.
    """
    eta, w, h, n = args
    (
        isLengthUnknown,
        isSourceUnknown,
        isConcUnknown,
    ) = variable_setup(whatToSolve)

    inc_mat = inc_mat_produce(inc_csv, address)

    arg = (w, h, n)
    A = construct_block_mat(length, eta, arg, inc_mat)

    b_f = construct_RHS(inc_mat, ini_list)

    ini_list_outlet = []
    for i in ini_list:
        if i[3] == "o":
            ini_list_outlet.append(i)

    ini_list_inlet = []
    for i in ini_list:
        if i[3] == "i":
            ini_list_inlet.append(i)

    A_ground, b_f_ground, gnd_idx = ground_block_mat(A, inc_mat, b_f, ini_list_outlet)
    i_vec, p_vec, b_f = Kirchhoff_solver(
        A,
        A_ground,
        b_f,
        b_f_ground,
        inc_mat,
        isSourceUnknown,
        ini_list,
        ini_list_outlet,
        ini_list_inlet,
        gnd_idx,
    )

    inlet_current = 0
    inlet_current_vec = -1 * i_vec[-len(ini_list_inlet) :]
    for i in inlet_current_vec:
        inlet_current += i

    return inlet_current


def process_optimization_results(res, callback, x0, address):
    """
    Process and visualize the optimization results.

    This function creates various plots to visualize the optimization results,
    including the Pareto front, convergence plots, and scatter plots of objectives.

    Args:
        res (Result): The result object from the optimization.
        callback (ConvergenceCallback): The callback object used during optimization.
        x0 (np.ndarray): The initial solution.
        address (str): The directory to save the output files.

    Returns:
        None
    """
    plt.rcParams.update({
        'font.size': 50,
        'text.usetex': True,
        'font.family': 'times',
        'axes.linewidth': 5,
        'lines.linewidth': 5,
        'xtick.major.width':5,
        'ytick.major.width':5,
        'xtick.major.size':10,
        'ytick.major.size':10,
        'xtick.major.pad':10,
        'ytick.major.pad':10,

        'xtick.minor.width':5,
        'ytick.minor.width':5,
        'xtick.minor.size':10,
        'ytick.minor.size':10,
    
        'legend.facecolor':'white',
        'legend.edgecolor':'None',
        'legend.framealpha':0.5
    })
    
    # Stack all populations into one array
    F = np.vstack(callback.data["F"])
    X = np.vstack(callback.data["X"])

    # Calculate distances from initial solution
    distances = np.linalg.norm(X - x0, axis=1)

    # Extract the best values for each objective at each iteration
    best_f1 = [np.min([f[0] for f in opt if np.isfinite(f[0])]) for opt in callback.opt]
    best_f2 = [np.min([f[1] for f in opt if np.isfinite(f[1])]) for opt in callback.opt]

    # Flow Diff vs Conc Diff
    res_F = res.F
    res_X = res.X
    plt.figure(figsize=(15, 12))
    plt.title("Flow Diff vs Conc Diff")
    plt.scatter(F[:, 0], F[:, 1], alpha=0.5, s=200)
    plt.scatter(res_F[:, 0], res_F[:, 1], alpha=0.5, color="red", s=200)
    plt.xlabel("Flow Diff")
    plt.ylabel("Conc Diff")
    plt.tight_layout()
    plt.savefig(f"{address}/flow_diff_vs_conc_diff.png")
    plt.close()

    # Flow Diff vs Distance
    res_distances = np.linalg.norm(res_X - x0, axis=1)
    plt.figure(figsize=(15, 12))
    plt.title("Flow Diff vs Distance")
    plt.scatter(F[:, 0], distances, alpha=0.5, s=200)
    plt.scatter(res_F[:, 0], res_distances, alpha=0.5, color="red", s=200)
    plt.xlabel("Flow Diff")
    plt.ylabel("Distance to x0")
    plt.tight_layout()
    plt.savefig(f"{address}/flow_diff_vs_distance.png")
    plt.close()

    # Conc Diff vs Distance
    plt.figure(figsize=(15, 12))
    plt.title("Conc Diff vs Distance")
    plt.scatter(F[:, 1], distances, alpha=0.5, s=200)
    plt.scatter(res_F[:, 1], res_distances, alpha=0.5, color="red", s=200)
    plt.xlabel("Conc Diff")
    plt.ylabel("Distance to x0")
    plt.tight_layout()
    plt.savefig(f"{address}/conc_diff_vs_distance.png")
    plt.close()

    # Plot the convergence for each objective
    for idx in range(2):
        plt.figure(figsize=(15, 12))
        plt.title(f"Convergence of Objective {idx+1}")
        plt.plot(
            callback.n_evals,
            best_f1 if idx == 0 else best_f2,
            label=f"Objective {idx+1}",
        )
        plt.xlabel("Number of Function Evaluations")
        plt.ylabel("Objective Value")
        plt.yscale("log")  # Use log scale if values span multiple orders of magnitude
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{address}/convergence_obj{idx+1}.png")
        plt.close()

    # Plot Pareto front
    plt.figure(figsize=(15, 12))
    plt.title("Flow Diff vs Conc Diff")
    plt.scatter(res_F[:, 0], res_F[:, 1], alpha=0.5, color="red", s=200)
    plt.xlabel("Flow Diff")
    plt.ylabel("Conc Diff")
    plt.tight_layout()
    plt.savefig(f"{address}/pareto_front.png")
    plt.close()

    # Save Pareto-optimal solutions and objectives
    pd.DataFrame(res.X).to_csv(f"{address}/pareto_optimal_solutions.csv")
    pd.DataFrame(res.F).to_csv(f"{address}/pareto_optimal_objectives.csv")

    print("Optimization results processed and saved.")


def select_optimal_solution(res):
    """
    Select the optimal solution from the Pareto front using the knee point method.

    This function normalizes the objectives, calculates the distances to the ideal point,
    and selects the solution with the maximum distance (knee point).

    Args:
        res (Result): The result object from the optimization algorithm.

    Returns:
        selected_solution (np.ndarray): The selected optimal solution.
    """
    # Normalize the objectives
    min_values = np.min(res.F, axis=0)
    max_values = np.max(res.F, axis=0)
    objective_span = max_values - min_values
    objective_span[objective_span == 0] = 1.0
    normalized_objectives = (res.F - min_values) / objective_span

    # Calculate distances to the ideal point (origin)
    distances = cdist(normalized_objectives, np.array([[0, 0]]))

    # Find the knee point (solution with maximum distance)
    knee_idx = np.argmax(distances)

    # Select the solution corresponding to the knee point
    selected_solution = res.X[knee_idx]

    return selected_solution


def constraint_diff(
    constraint,
    x,
    whatToSolve,
    changing_edges,
    inc_csv,
    length_csv,
    args,
    ini_list,
    inlet_node_idx,
    inlet_edge_idx,
    address,
    popSize,
):
    """
    Calculate the difference between constraints and actual values for optimization.

    This function computes the difference between the given constraints and the
    actual values obtained from the length_change_func for flow rates and concentrations.

    Args:
        constraint (list): List of constraint values for flow rates and concentrations.
        x (np.ndarray): Array of decision variables (lengths).
        whatToSolve (str or list): Indicates what to solve in the problem.
        changing_edges (list): List of edges that can be changed.
        inc_csv (str): Filename of the incidence matrix CSV.
        length_csv (str): Filename of the length matrix CSV.
        args (tuple): Additional arguments for the problem.
        ini_list (list): Initial list of values.
        inlet_node_idx (list): Indices of inlet nodes.
        inlet_edge_idx (list): Indices of inlet edges.
        address (str): Directory path for saving output files.
        popSize (int): Population size for vectorized calculations.

    Returns:
        tuple: Contains the following:
            - Difference between constraint and actual flow rates.
            - Difference between constraint and actual concentrations.
            - Actual flow rates.
            - Actual concentrations.
    """
    result = []
    value = []

    # Calculate actual values using length_change_func
    actual_values = length_change_func(
        x,
        whatToSolve,
        changing_edges,
        inc_csv,
        length_csv,
        args,
        ini_list,
        inlet_node_idx,
        inlet_edge_idx,
        address,
        popSize,
    )

    # Calculate differences for flow rates and concentrations
    for idx, const in enumerate(constraint):
        tmp = actual_values[idx]
        res = np.abs(const - tmp)
        result.append(res)
        value.append(tmp)

    # Combine results and values
    result.extend(value)
    return tuple(result)


def length_change_func(
    unks,
    whatToSolve,
    changing_edges,
    inc_csv,
    length_csv,
    args,
    ini_list,
    inlet_node_idx,
    inlet_edge_idx,
    address,
    popSize,
):
    """
    Calculate the flow and concentration results after changing the lengths of specified edges.

    This function modifies the lengths of specified edges, solves the microfluidic circuit,
    and returns the scaled outlet flow rates and concentrations.

    Args:
        unks (np.ndarray): Array of new lengths for the changing edges.
        whatToSolve (str or list): Indicates what to solve in the problem.
        changing_edges (list): Indices of edges whose lengths are being changed.
        inc_csv (str): Filename of the incidence matrix CSV.
        length_csv (str): Filename of the length matrix CSV.
        args (tuple): Contains (eta, w, h, n) where:
            eta (float): Fluid viscosity.
            w (float): Channel width.
            h (float): Channel height.
            n (int): Computation coefficient.
        ini_list (list): Initial values of voltage & current sources.
        inlet_node_idx (list): Indices of inlet nodes.
        inlet_edge_idx (list): Indices of inlet edges.
        address (str): Directory path for saving output files.
        popSize (int): Population size for vectorized calculations.

    Returns:
        result (list): Contains two numpy arrays:
            - Scaled outlet flow rates
            - Outlet concentrations (if applicable)
    """
    eta, w, h, n = args
    (
        isLengthUnknown,
        isSourceUnknown,
        isConcUnknown,
    ) = variable_setup(whatToSolve)

    # Load and prepare matrices
    inc_mat = inc_mat_produce(inc_csv, address)
    inc_mat_vec = np.broadcast_to(inc_mat, (popSize,) + inc_mat.shape)

    length_df = length_mat_produce(length_csv, address)
    length = np.array(list(length_df["length"]))
    length_vec = np.broadcast_to(length, (popSize,) + length.shape)
    length_vec_copy = np.copy(length_vec)

    # Update lengths of changing edges
    changing_edges_mask = np.isin(np.arange(length_vec_copy.shape[1]), changing_edges)
    length_vec_copy[:, changing_edges_mask] = unks[
        :, np.arange(changing_edges_mask.sum())
    ]

    # Prepare lists and matrices
    ini_list_outlet = []
    for i in ini_list:
        if i[3] == "o":
            ini_list_outlet.append(i)

    ini_list_inlet = []
    for i in ini_list:
        if i[3] == "i":
            ini_list_inlet.append(i)

    arg = (w, h, n)
    A_vec = construct_block_mat_vec(length_vec_copy, eta, arg, inc_mat_vec, popSize)

    # Construct RHS and ground matrices
    b_f = construct_RHS(inc_mat, ini_list)
    b_f_vec = np.broadcast_to(b_f, (popSize,) + b_f.shape)

    A_ground_vec, b_f_ground_vec, gnd_idx = ground_block_mat_vec(
        A_vec, inc_mat_vec, b_f_vec, ini_list_outlet
    )

    # Solve the circuit
    # try:
    i_vec, p_vec, b_f_vec = Kirchhoff_solver_vec(
        A_vec,
        A_ground_vec,
        b_f_vec,
        b_f_ground_vec,
        inc_mat_vec,
        isSourceUnknown,
        ini_list,
        ini_list_outlet,
        ini_list_inlet,
        gnd_idx,
        popSize,
    )

    # Calculate concentrations if required
    conc_vec = None

    if isConcUnknown and len(ini_list_inlet) > 1:
        conc_vec = conc_calculation_vec(
            b_f_vec,
            i_vec,
            inc_mat_vec,
            ini_list_inlet,
            ini_list_outlet,
            inlet_node_idx,
            inlet_edge_idx,
            popSize,
        )

    conc_vec_total = np.concatenate(tuple(conc_vec), axis=1)

    # Calculate scaled outlet flow rates
    i_vec_outlet = i_vec[:, : len(ini_list_outlet)]

    total_flow_rate = 0
    current_driven = False
    for i in ini_list_inlet:
        if i[4] == "m^3/s":
            current__driven = True
            total_flow_rate += i[2]

    if current_driven:
        total_flow_rate = total_flow_rate * -1 * 10**6 * 60

    else:
        A = construct_block_mat(length, eta, arg, inc_mat)

        b_f = construct_RHS(inc_mat, ini_list)

        A_ground, b_f_ground, gnd_idx = ground_block_mat(
            A, inc_mat, b_f, ini_list_outlet
        )
        i, p, b_f = Kirchhoff_solver(
            A,
            A_ground,
            b_f,
            b_f_ground,
            inc_mat,
            isSourceUnknown,
            ini_list,
            ini_list_outlet,
            ini_list_inlet,
            gnd_idx,
        )
        inlet_current_vec = i[-len(ini_list_inlet) :]
        for idx in inlet_current_vec:
            total_flow_rate += idx

    i_vec_outlet_scaled = i_vec_outlet / total_flow_rate

    result = [i_vec_outlet_scaled, conc_vec_total if conc_vec is not None else None]

    return result

    # except:
    #     return [100, 100]


def opt_time_acc(
    whatToSolve,
    changing_edges,
    inc_csv,
    length_csv,
    conc_csv,
    args,
    ini_list,
    inlet_node_idx,
    inlet_edge_idx,
    outlet_node_idx,
    outlet_edge_idx,
    conc_wght,
    flow_wght,
    address,
    popSize,
    nGen,
):
    """
    Perform optimization and calculate time and accuracy metrics.

    This function executes length change optimization and calculates the objective
    values and time taken for the optimization process.

    Args:
        whatToSolve (str or list): Indicates what to solve in the problem.
        changing_edges (list): List of edges that can be changed.
        inc_csv (str): Filename of the incidence matrix CSV.
        length_csv (str): Filename of the length matrix CSV.
        conc_csv (str): Filename of the concentration matrix CSV.
        args (tuple): Contains (eta, w, h, n) where:
            eta (float): Fluid viscosity.
            w (float): Channel width.
            h (float): Channel height.
            n (int): Computation coefficient.
        ini_list (list): Initial values of voltage & current sources.
        inlet_node_idx (list): Indices of inlet nodes.
        inlet_edge_idx (list): Indices of inlet edges.
        outlet_node_idx (list): Indices of outlet nodes.
        outlet_edge_idx (list): Indices of outlet edges.
        conc_wght (float): Weight for concentration in the objective function.
        flow_wght (float): Weight for flow in the objective function.
        address (str): Directory path for saving output files.
        popSize (int): Population size for optimization.
        nGen (int): Number of generations for optimization.

    Returns:
        tuple: Contains the following:
            objective (list): [flow_diff_norm, conc_diff_norm]
            time_diff (float): Time taken for optimization.
    """
    # Execute length change optimization
    popSize = int(popSize)
    nGen = int(nGen)
    
    time_diff, length_df = execute_length_change(
        whatToSolve,
        changing_edges,
        inc_csv,
        length_csv,
        conc_csv,
        args,
        ini_list,
        inlet_node_idx,
        inlet_edge_idx,
        outlet_node_idx,
        outlet_edge_idx,
        conc_wght,
        flow_wght,
        address,
        popSize,
        nGen,
    )

    # Unpack arguments and setup variables
    eta, w, h, n = args
    (
        isLengthUnknown,
        isSourceUnknown,
        isConcUnknown,
    ) = variable_setup(whatToSolve)

    # Produce matrices
    inc_mat = inc_mat_produce(inc_csv, address)
    length = np.array(list(length_df["length"]))
    length_vec = np.broadcast_to(length, (popSize,) + length.shape)

    # Prepare lists for inlet and outlet
    ini_list_outlet = []
    for i in ini_list:
        if i[3] == "o":
            ini_list_outlet.append(i)

    ini_list_inlet = []
    for i in ini_list:
        if i[3] == "i":
            ini_list_inlet.append(i)

    # Calculate inlet current and target concentrations
    (
        inlet_current,
        outlet_current,
        outlet_current_vec,
        outlet_current_vec_scaled,
    ) = calculate_currents(whatToSolve, inc_csv, length, ini_list, args, address)

    tmp_conc_df = conc_mat_produce(conc_csv, address)
    target_conc_df = []
    tmp_conc_df_col = list(tmp_conc_df.columns)
    for i in range(len(tmp_conc_df.columns)):
        tmp_col = copy.deepcopy(tmp_conc_df_col)
        tmp_col.remove(f"fraction{i+1}")
        tmp = tmp_conc_df.drop(columns=tmp_col)
        target_conc_df.append(tmp)

    target_conc_vec = [i.to_numpy() for i in target_conc_df]
    target_conc_vec_total = np.vstack(tuple(target_conc_vec)).T.flatten()

    constraint = [outlet_current_vec_scaled, target_conc_vec_total]

    # Calculate differences
    flow_diff, conc_diff, flow, conc = constraint_diff(
        constraint,
        length_vec,
        whatToSolve,
        changing_edges,
        inc_csv,
        length_csv,
        args,
        ini_list,
        inlet_node_idx,
        inlet_edge_idx,
        address,
        popSize,
    )

    # Calculate objective values
    objective = [np.linalg.norm(flow_diff), np.linalg.norm(conc_diff)]

    return objective, time_diff


def pop_gen_opt(
    whatToSolve,
    changing_edges,
    inc_csv,
    length_csv,
    conc_csv,
    args,
    ini_list,
    inlet_node_idx,
    inlet_edge_idx,
    outlet_node_idx,
    outlet_edge_idx,
    conc_wght,
    flow_wght,
    address,
):
    """
    Optimize population size and number of generations for the genetic algorithm.

    This function uses a multi-objective optimization approach to find the best
    combination of population size and number of generations for the genetic algorithm.

    Args:
        whatToSolve (str or list): Indicates what to solve in the problem.
        changing_edges (list): List of edges that can be changed.
        inc_csv (str): Filename of the incidence matrix CSV.
        length_csv (str): Filename of the length matrix CSV.
        conc_csv (str): Filename of the concentration matrix CSV.
        args (tuple): Contains (eta, w, h, n) where:
            eta (float): Fluid viscosity.
            w (float): Channel width.
            h (float): Channel height.
            n (int): Computation coefficient.
        ini_list (list): Initial values of voltage & current sources.
        inlet_node_idx (list): Indices of inlet nodes.
        inlet_edge_idx (list): Indices of inlet edges.
        outlet_node_idx (list): Indices of outlet nodes.
        outlet_edge_idx (list): Indices of outlet edges.
        conc_wght (float): Weight for concentration in the objective function.
        flow_wght (float): Weight for flow in the objective function.
        address (str): Directory path for saving output files.

    Returns:
        None. Results are saved to CSV files.
    """
    # Set bounds for population size and number of generations
    lb = [20, 20]
    ub = [400, 400]

    # Create the optimization problem
    problem = MicrofluidicOptimalpointFinding(
        whatToSolve,
        changing_edges,
        inc_csv,
        length_csv,
        conc_csv,
        args,
        ini_list,
        inlet_node_idx,
        inlet_edge_idx,
        outlet_node_idx,
        outlet_edge_idx,
        conc_wght,
        flow_wght,
        address,
        lb,
        ub,
    )

    # Set up the genetic algorithm
    algorithm = NSGA2(
        pop_size=100,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9),
        mutation=PolynomialMutation(prob=0.1),
        eliminate_duplicates=True,
    )

    # Run the optimization
    res = minimize(problem, algorithm, ("n_gen", 100), seed=1, verbose=True)

    # Save results to CSV files
    pd.DataFrame(res.X).to_csv(f"{address}/timeacc_optimal_solutions.csv")
    pd.DataFrame(res.F).to_csv(f"{address}/timeacc_optimal_objectives.csv")

    print("Optimization complete. Results saved to CSV files.")


def inlet_computing(
    whatToSolve,
    inc_csv,
    length_csv,
    args,
    ini_list,
    inlet_node_idx,
    inlet_edge_idx,
    address,
    target_conc,
    target_outlet,
):
    """
    Compute optimal inlet flow rates based on target concentration at a specific outlet.

    This function chooses a target outlet and target concentration, then finds
    the optimal inlet flow rates to achieve the desired concentration at the target outlet.

    The function constructs an optimization problem where the domain is the inlet flow rates
    and the range is the outlet concentrations at the target outlet. It then solves this
    problem to find the optimal inlet flow rates that achieve the target concentration.

    Args:
        whatToSolve (str or list): Indicates what to solve in the problem.
        inc_csv (str): Filename of the incidence matrix CSV.
        length_csv (str): Filename of the length matrix CSV.
        args (tuple): Additional arguments for the problem.
        ini_list (list): Initial list of values.
        inlet_node_idx (list): Indices of inlet nodes.
        inlet_edge_idx (list): Indices of inlet edges.
        address (str): File path for saving results.
        target_conc (list): Target concentrations for each inlet.
        target_outlet (int): Index of the target outlet.

    Returns:
        tuple: Contains the following:
            inlet_pressure (np.array): Computed inlet pressures.
            inlet_flow_rate (np.array): Computed inlet flow rates.
            conc_vec (list or None): Concentration vectors if calculated, else None.

    Note:
        The number of target concentrations is the same as the number of inlet flow rates.
        The target concentrations have a constraint that their sum must be 1.
        The total flow rate is given to solve the problem appropriately.
        This additional constraint on the total flow rate matches the additional
        constraint on the sum of concentrations, making the problem well-defined.
    """
    # Extract original inlet flow rates
    original_inlet = []
    for i in ini_list:
        if i[0] == "f":
            if i[3] == "i":
                original_inlet.append(i[2])
    original_inlet = np.array(original_inlet)
    total_flow_rate = np.sum(original_inlet)

    # Define optimization function
    optimization_func = lambda x: target_conc_computing(
        x,
        whatToSolve,
        inc_csv,
        length_csv,
        args,
        ini_list,
        inlet_node_idx,
        inlet_edge_idx,
        address,
        target_conc,
        target_outlet,
        total_flow_rate,
    )

    # Solve for optimal inlet flow rates
    soln = optimize.root(optimization_func, original_inlet)
    revised_inlet = soln.x

    # Update ini_list with revised inlet flow rates
    for i, inlet in enumerate(ini_list):
        if inlet[0] == "f" and inlet[3] == "i":
            inlet[2] = revised_inlet[i]

    # Solve the circuit with revised inlet flow rates
    eta, w, h, n = args
    (
        isLengthUnknown,
        isSourceUnknown,
        isConcUnknown,
    ) = variable_setup(whatToSolve)

    inc_mat = inc_mat_produce(inc_csv, address)

    length_df = length_mat_produce(length_csv, address)
    length = list(length_df["length"])

    ini_list_outlet = []
    for i in ini_list:
        if i[3] == "o":
            ini_list_outlet.append(i)

    arg = (w, h, n)
    A = construct_block_mat(length, eta, arg, inc_mat)

    b_f = construct_RHS(inc_mat, ini_list)

    A_ground, b_f_ground, gnd_idx = ground_block_mat(A, inc_mat, b_f, ini_list_outlet)

    conc_vec = None
    ini_list_inlet = []
    for i in ini_list:
        if i[3] == "i":
            ini_list_inlet.append(i)

    i_vec, p_vec, b_f = Kirchhoff_solver(
        A,
        A_ground,
        b_f,
        b_f_ground,
        inc_mat,
        isSourceUnknown,
        ini_list,
        ini_list_outlet,
        ini_list_inlet,
        gnd_idx,
    )

    if isConcUnknown and len(ini_list_inlet) > 1:
        conc_vec = conc_calculation(
            b_f,
            i_vec,
            inc_mat,
            ini_list_inlet,
            ini_list_outlet,
            inlet_node_idx,
            inlet_edge_idx,
        )

    inlet_pressure = p_vec[-len(ini_list_inlet) :]
    inlet_flow_rate = i_vec[-len(ini_list_inlet) :]

    return inlet_pressure, inlet_flow_rate, conc_vec


def target_conc_computing(
    unks,
    whatToSolve,
    inc_csv,
    length_csv,
    args,
    ini_list,
    inlet_node_idx,
    inlet_edge_idx,
    address,
    target_conc,
    target_outlet,
    total_flow_rate,
):
    """
    Compute the difference between target and actual concentrations for given inlet flow rates.

    This function calculates the concentration at the target outlet based on the given
    inlet flow rates and compares it with the target concentration.

    Args:
        unks (np.array): Unknown inlet flow rates.
        whatToSolve (str or list): Indicates what to solve in the problem.
        inc_csv (str): Filename of the incidence matrix CSV.
        length_csv (str): Filename of the length matrix CSV.
        args (tuple): Contains (eta, w, h, n) where:
            eta (float): Fluid viscosity.
            w (float): Channel width.
            h (float): Channel height.
            n (int): Computation coefficient.
        ini_list (list): Initial values of voltage & current sources.
        inlet_node_idx (list): Indices of inlet nodes.
        inlet_edge_idx (list): Indices of inlet edges.
        address (str): Directory path for saving output files.
        target_conc (np.array): Target concentrations for each inlet.
        target_outlet (int): Index of the target outlet.
        total_flow_rate (float): Total flow rate of the system.

    Returns:
        diff (np.array): Difference between target and computed concentrations.
    """
    eta, w, h, n = args
    (
        isLengthUnknown,
        isSourceUnknown,
        isConcUnknown,
    ) = variable_setup(whatToSolve)

    # Load and prepare matrices
    inc_mat = inc_mat_produce(inc_csv, address)
    length_df = length_mat_produce(length_csv, address)
    length = list(length_df["length"])

    # Update ini_list with new inlet flow rates
    cnt = 0
    for i in ini_list:
        if i[0] == "f":
            if i[3] == "i":
                if cnt < len(unks):
                    i[2] = unks[cnt]
                    cnt += 1
                elif cnt == len(unks):
                    i[2] = total_flow_rate - np.sum(unks)
                    cnt += 1

    # Prepare lists for inlet and outlet
    ini_list_outlet = []
    for i in ini_list:
        if i[3] == "o":
            ini_list_outlet.append(i)

    ini_list_inlet = []
    for i in ini_list:
        if i[3] == "i":
            ini_list_inlet.append(i)

    print("ini_list: \n", ini_list)

    # Construct block matrix and solve Kirchhoff's equations
    arg = (w, h, n)
    A = construct_block_mat(length, eta, arg, inc_mat)

    b_f = construct_RHS(inc_mat, ini_list)

    A_ground, b_f_ground, gnd_idx = ground_block_mat(A, inc_mat, b_f, ini_list_outlet)

    i_vec, p_vec, b_f = Kirchhoff_solver(
        A,
        A_ground,
        b_f,
        b_f_ground,
        inc_mat,
        isSourceUnknown,
        ini_list,
        ini_list_outlet,
        ini_list_inlet,
        gnd_idx,
    )

    # Calculate concentrations if required
    conc_vec = None
    if isConcUnknown and len(ini_list_inlet) > 1:
        conc_vec = conc_calculation(
            b_f,
            i_vec,
            inc_mat,
            ini_list_inlet,
            ini_list_outlet,
            inlet_node_idx,
            inlet_edge_idx,
        )

    # Compare computed concentrations with target concentrations
    computed_conc = []
    for i in conc_vec:
        computed_conc.append(i[target_outlet])

    computed_conc = np.array(computed_conc)
    print("computed conc: ", computed_conc)
    diff = np.abs(target_conc - computed_conc)

    print("diff: ", diff)

    diff = diff[: len(ini_list_inlet)]

    print("diff: ", diff)

    return diff


def analyze_and_select_solution_timeacc(
    all_solutions,
    all_objectives,
    address,
):
    """
    Analyze Pareto front solutions and select the best solutions using different methods.

    This function plots the Pareto front for different objective combinations,
    and selects the best solutions using three different methods:
    1. Closest to the ideal point
    2. Minimum sum of normalized objectives
    3. Knee point of the Pareto front

    Args:
        all_solutions (np.ndarray): Array of all solutions.
        all_objectives (np.ndarray): Array of corresponding objective values.
        address (str): Directory path for saving the plots.

    Returns:
        tuple: Indices of the best solutions for each method
            (best_idx, best_idx_sum, knee_idx)
    """
    # Plot Pareto front
    plot_pareto_front(
        all_objectives[:, 0],
        all_objectives[:, 1],
        "Flow Difference",
        "Concentration Difference",
        f"{address}/pareto_front_flow_conc.png",
    )

    plot_pareto_front(
        all_objectives[:, 0],
        all_objectives[:, 2],
        "Flow Difference",
        "Time",
        f"{address}/pareto_front_flow_time.png",
    )

    plot_pareto_front(
        all_objectives[:, 1],
        all_objectives[:, 2],
        "Concentration Difference",
        "Time",
        f"{address}/pareto_front_conc_time.png",
    )

    # Method 1: Select the solution closest to the ideal point
    best_idx = select_closest_to_ideal(all_objectives)
    best_solution_ideal = all_solutions[best_idx]

    # Method 2: Select the solution with the minimum sum of normalized objectives
    best_idx_sum = select_min_sum_normalized(all_objectives)
    best_solution_sum = all_solutions[best_idx_sum]

    # Method 3: Select the knee point of the Pareto front
    knee_idx = select_knee_point(all_objectives)
    best_solution_knee = all_solutions[knee_idx]

    print("Best solution (closest to ideal point):", best_solution_ideal)
    print("Best solution (minimum sum of normalized objectives):", best_solution_sum)
    print("Best solution (knee point):", best_solution_knee)

    print("Best solution index (closest to ideal point):", best_idx)
    print("Best solution index (minimum sum of normalized objectives):", best_idx_sum)
    print("Best solution index (knee point):", knee_idx)

    return best_idx, best_idx_sum, knee_idx


def plot_pareto_front(x, y, xlabel, ylabel, filename):
    """
    Plot and save a 2D Pareto front.

    This function creates a scatter plot of the Pareto front and saves it to a file.

    Args:
        x (np.ndarray): x-coordinates of the Pareto front points.
        y (np.ndarray): y-coordinates of the Pareto front points.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        filename (str): Path where the plot will be saved.

    Returns:
        None
    """
    plt.rcParams.update({
        'font.size': 50,
        'text.usetex': True,
        'font.family': 'times',
        'axes.linewidth': 5,
        'lines.linewidth': 5,
        'xtick.major.width':5,
        'ytick.major.width':5,
        'xtick.major.size':10,
        'ytick.major.size':10,
        'xtick.major.pad':10,
        'ytick.major.pad':10,

        'xtick.minor.width':5,
        'ytick.minor.width':5,
        'xtick.minor.size':10,
        'ytick.minor.size':10,
    
        'legend.facecolor':'white',
        'legend.edgecolor':'None',
        'legend.framealpha':0.5
    })
    
    plt.figure(figsize=(15, 12))
    plt.scatter(x, y, c="blue", s=200)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"Pareto Front ({xlabel} vs {ylabel})")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def select_closest_to_ideal(objectives):
    """
    Select the solution closest to the ideal point.

    This function finds the solution that has the minimum Euclidean distance
    to the ideal point (minimum of each objective).

    Args:
        objectives (np.ndarray): Array of objective values for each solution.
            Shape: (n_solutions, n_objectives)

    Returns:
        int: Index of the solution closest to the ideal point.
    """
    ideal_point = np.min(objectives, axis=0)
    distances = cdist(objectives, [ideal_point])
    return np.argmin(distances)


def select_min_sum_normalized(objectives):
    """
    Select the solution with the minimum sum of normalized objectives.

    This function normalizes the objectives and then selects the solution
    with the minimum sum of these normalized values.

    Args:
        objectives (np.ndarray): Array of objective values for each solution.
            Shape: (n_solutions, n_objectives)

    Returns:
        int: Index of the solution with the minimum sum of normalized objectives.
    """
    normalized_objectives = normalize_objectives(objectives)
    sum_normalized = np.sum(normalized_objectives, axis=1)
    return np.argmin(sum_normalized)


def select_knee_point(objectives):
    """
    Select the knee point of the Pareto front.

    This function identifies the knee point, which is the solution with the
    maximum distance from the line connecting the extreme solutions in the
    normalized objective space.

    Args:
        objectives (np.ndarray): Array of objective values for each solution.
            Shape: (n_solutions, n_objectives)

    Returns:
        int: Index of the knee point solution.
    """
    normalized_objectives = normalize_objectives(objectives)
    distances = cdist(normalized_objectives, np.array([[0, 0, 0]]))
    return np.argmax(distances)


def normalize_objectives(objectives):
    """
    Normalize the objectives to the range [0, 1].

    This function performs min-max normalization on the objectives.

    Args:
        objectives (np.ndarray): Array of objective values for each solution.
            Shape: (n_solutions, n_objectives)

    Returns:
        np.ndarray: Normalized objective values.
            Shape: (n_solutions, n_objectives)
    """
    return (objectives - np.min(objectives, axis=0)) / (
        np.max(objectives, axis=0) - np.min(objectives, axis=0)
    )


def optimization_functions(
    whatToSolve,
    changing_edges,
    inc_csv,
    length_csv,
    conc_csv,
    args,
    ini_list,
    inlet_node_idx,
    inlet_edge_idx,
    outlet_node_idx,
    outlet_edge_idx,
    conc_wght,
    flow_wght,
    plot_type,
    address,
):
    """
    Perform optimization and analysis for the given problem.
    
    This function executes the optimization process for the population size and the number of generations,
    and analyzes the results
    
    Args:
        whatToSolve (str or list): Indicates what to solve in the problem.
        changing_edges (list): List of edges that can be changed.
        inc_csv (str): Filename of the incidence matrix CSV.
        length_csv (str): Filename of the length matrix CSV.
        conc_csv (str): Filename of the concentration matrix CSV.
        args (tuple): Contains (eta, w, h, n) where:
            eta (float): Fluid viscosity.
            w (float): Channel width.
            h (float): Channel height.
            n (int): Computation coefficient.
        ini_list (list): Initial values of voltage & current sources.
        inlet_node_idx (list): Indices of inlet nodes.
        inlet_edge_idx (list): Indices of inlet edges.
        outlet_node_idx (list): Indices of outlet nodes.
        outlet_edge_idx (list): Indices of outlet edges.
        conc_wght (float): Weight for concentration in the objective function.
        flow_wght (float): Weight for flow in the objective function.
        plot_type (str): Type of plot to generate.
        address (str): Directory path for saving output files.
    
    Returns:
        None
    """
    if "length" in whatToSolve:
        pop_gen_opt(
            whatToSolve,
            changing_edges,
            inc_csv,
            length_csv,
            conc_csv,
            args,
            ini_list,
            inlet_node_idx,
            inlet_edge_idx,
            outlet_node_idx,
            outlet_edge_idx,
            conc_wght,
            flow_wght,
            address,
        )

        timeacc_solutions = (
            pd.read_csv(f"{address}/timeacc_optimal_solutions.csv")
            .drop(labels="Unnamed: 0", axis=1)
            .values
        )
        timeacc_objectives = (
            pd.read_csv(f"{address}/timeacc_optimal_objectives.csv")
            .drop(labels="Unnamed: 0", axis=1)
            .values
        )

        (
            best_idx_timeacc,
            best_idx_sum_timeacc,
            knee_idx_timeacc,
        ) = analyze_and_select_solution_timeacc(
            timeacc_solutions, timeacc_objectives, address
        )

        popSize_ideal, nGen_ideal = tuple(timeacc_solutions[best_idx_timeacc])
        popSize_sum, nGen_sum = tuple(timeacc_solutions[best_idx_sum_timeacc])
        popSize_knee, nGen_knee = tuple(timeacc_solutions[knee_idx_timeacc])

        ideal = [popSize_ideal, nGen_ideal]
        sum = [popSize_sum, nGen_sum]
        knee = [popSize_knee, nGen_knee]

        opt_coeffs = [ideal, sum, knee]
        nametagList = ["ideal", "sum", "knee"]

        for i in range(len(nametagList)):
            execute_length_change(
                whatToSolve,
                changing_edges,
                inc_csv,
                length_csv,
                conc_csv,
                args,
                ini_list,
                inlet_node_idx,
                inlet_edge_idx,
                outlet_node_idx,
                outlet_edge_idx,
                conc_wght,
                flow_wght,
                address,
                opt_coeffs[i][0],
                opt_coeffs[i][1],
            )
            nametag = nametagList[i]
            length_csv = f"new_length_mat_{nametag}.csv"

            i_vec, conc_vec = execute_functions(
                whatToSolve,
                inc_csv,
                length_csv,
                args,
                ini_list,
                inlet_node_idx,
                inlet_edge_idx,
                nametag,
                address,
            )

            plotting_outlet(
                i_vec,
                outlet_edge_idx,
                plot_type,
                address,
                whatToSolve,
                ini_list,
                args,
                conc_vec,
                nametag,
                inc_csv,
                length_csv,
                conc_csv,
            )
            
            
def constraint_diff_pareto(
    constraint,
    x,
    whatToSolve,
    inc_csv,
    args,
    ini_list,
    inlet_node_idx,
    inlet_edge_idx,
    address,
    popSize,
):
    """Modified constraint_diff function for Pareto comparison."""
    result = []
    value = []

    # Calculate actual values using length_change_func_pareto
    actual_values = length_change_func_pareto(
        x,
        whatToSolve,
        inc_csv,
        args,
        ini_list,
        inlet_node_idx,
        inlet_edge_idx,
        address,
        popSize,
    )

    # Calculate differences for flow rates and concentrations
    for idx, const in enumerate(constraint):
        tmp = actual_values[idx]
        res = np.abs(const - tmp)
        result.append(res)
        value.append(tmp)

    # Combine results and values
    result.extend(value)
    return tuple(result)

def pareto_compare(
    whatToSolve,
    inc_csv,
    length_csv,
    conc_csv,
    args,
    ini_list,
    inlet_node_idx,
    inlet_edge_idx,
    outlet_node_idx,
    outlet_edge_idx,
    address,
):
    """
    Compare and validate Pareto front solutions by analyzing their performance.
    Handles CSV format with 26 edge columns (e1-e26).
    """
    plt.rcParams.update({
        'font.size': 50,
        'text.usetex': True,
        'font.family': 'times',
        'axes.linewidth': 5,
        'lines.linewidth': 5,
        'xtick.major.width':5,
        'ytick.major.width':5,
        'xtick.major.size':10,
        'ytick.major.size':10,
        'xtick.major.pad':10,
        'ytick.major.pad':10,

        'xtick.minor.width':5,
        'ytick.minor.width':5,
        'xtick.minor.size':10,
        'ytick.minor.size':10,
    
        'legend.facecolor':'white',
        'legend.edgecolor':'None',
        'legend.framealpha':0.5
    })
    
    # Read all Pareto front solutions
    pareto_lengths = pd.read_csv(f"{address}/{length_csv}")
    # Select only the edge columns (e1 through e26)
    length_values = pareto_lengths[[f'e{i}' for i in range(1, inlet_edge_idx[-1] + 2)]].values
    
    print(f"Number of Pareto solutions: {len(length_values)}")
    print(f"Shape of length values: {length_values.shape}")
    
    # Calculate target flow rates and concentrations
    inlet_current, outlet_current, outlet_current_vec, outlet_current_vec_scaled = calculate_currents(
        whatToSolve, inc_csv, length_values[0], ini_list, args, address
    )
    
    # Read target concentrations
    tmp_conc_df = conc_mat_produce(conc_csv, address)
    target_conc_df = []
    tmp_conc_df_col = list(tmp_conc_df.columns)
    for i in range(len(tmp_conc_df.columns)):
        tmp_col = copy.deepcopy(tmp_conc_df_col)
        tmp_col.remove(f"fraction{i+1}")
        tmp = tmp_conc_df.drop(columns=tmp_col)
        target_conc_df.append(tmp)
        
    target_conc_vec = [i.to_numpy() for i in target_conc_df]
    target_conc_vec_total = np.vstack(tuple(target_conc_vec)).T.flatten()
    
    # Store results
    flow_diffs = []
    conc_diffs = []
    actual_flows = []
    actual_concs = []
    actual_conc_diffs = []
    
    print(f"\nProcessing {len(length_values)} solutions...")
    
    # Process each solution
    for idx, length in enumerate(length_values):
        try:
            # Reshape length to match expected dimensions (1, num_edges)
            length_vec = length.reshape(1, -1)
            
            flow_diff, conc_diff, flow, conc = constraint_diff_pareto(
                [outlet_current_vec_scaled, target_conc_vec_total],
                length_vec,
                whatToSolve,
                inc_csv,
                args,
                ini_list,
                inlet_node_idx,
                inlet_edge_idx,
                address,
                1  # popSize=1 since we're evaluating one solution at a time
            )
            
            actual_conc_diffs.append(conc_diff[0] if isinstance(conc_diff, np.ndarray) else conc_diff)
            
            # Extract scalar values
            if isinstance(flow_diff, np.ndarray):
                flow_diff = np.mean(flow_diff.ravel())
            if isinstance(conc_diff, np.ndarray):
                conc_diff = np.mean(conc_diff.ravel())
            
            flow_diffs.append(flow_diff)
            conc_diffs.append(conc_diff)
            actual_flows.append(flow[0] if isinstance(flow, np.ndarray) else flow)
            actual_concs.append(conc[0] if isinstance(conc, np.ndarray) else conc)
            
            if (idx + 1) % 5 == 0:
                print(f"Processed {idx + 1}/{len(length_values)} solutions")
                
        except Exception as e:
            print(f"Error processing solution {idx}: {str(e)}")
            continue
    
    # Convert to numpy arrays
    flow_diffs = np.array(flow_diffs)
    conc_diffs = np.array(conc_diffs)
    actual_flows = np.array(actual_flows)
    actual_concs = np.array(actual_concs)
    actual_conc_diffs = np.array(actual_conc_diffs)
    
    print(f"\nFinal array shapes:")
    print(f"flow_diffs shape: {flow_diffs.shape}")
    print(f"conc_diffs shape: {conc_diffs.shape}")
    print(f"actual_flows shape: {actual_flows.shape}")
    print(f"actual_concs shape: {actual_concs.shape}")
    
    if len(flow_diffs) == 0 or len(conc_diffs) == 0:
        print("No valid solutions to plot!")
        return
    
    # Create visualizations
    plt.figure(figsize=(15, 12))
    plt.scatter(flow_diffs, conc_diffs, alpha=0.5, s=200)
    plt.xlabel('Flow Rate Difference')
    plt.ylabel('Concentration Difference')
    plt.title('Flow vs Concentration Differences')
    plt.tight_layout()
    plt.savefig(f"{address}/pareto_comparison_differences.png")
    plt.close()
    
    if len(actual_flows) > 0 and len(actual_concs) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 12))
        
        ax1.boxplot(actual_flows)
        ax1.axhline(y=outlet_current_vec_scaled[0], color='r', linestyle='--', label='Target')
        ax1.set_title('Distribution of Flow Rates')
        ax1.set_ylabel('Flow Rate (scaled)')
        ax1.legend()
        
        ax2.boxplot(actual_conc_diffs)
        ax2.axhline(y=0, color='r', linestyle='--', label='Target')
        ax2.set_title('Distribution of Concentrations')
        ax2.set_ylabel('Concentration')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f"{address}/pareto_comparison_distributions.png")
        plt.close()
    
    # Calculate statistics
    if len(flow_diffs) > 0 and len(conc_diffs) > 0:
        stats = {
            'flow_diff_mean': np.mean(flow_diffs),
            'flow_diff_std': np.std(flow_diffs),
            'conc_diff_mean': np.mean(conc_diffs),
            'conc_diff_std': np.std(conc_diffs),
            'flow_target_mean_error': np.mean(np.abs(actual_flows - outlet_current_vec_scaled[0])),
            'conc_target_mean_error': np.mean(np.abs(actual_concs - target_conc_vec_total[0]))
        }
        
        # Save statistics
        pd.DataFrame([stats]).to_csv(f"{address}/pareto_comparison_stats.csv")
        
        # Find best solutions
        best_flow_idx = np.argmin(flow_diffs)
        best_conc_idx = np.argmin(conc_diffs)
        
        print("\nStatistics of Pareto front solutions:")
        print(f"Flow difference - Mean: {stats['flow_diff_mean']:.4f}, Std: {stats['flow_diff_std']:.4f}")
        print(f"Concentration difference - Mean: {stats['conc_diff_mean']:.4f}, Std: {stats['conc_diff_std']:.4f}")
        print(f"\nBest solution for flow (index {best_flow_idx}):")
        print(f"Flow difference: {flow_diffs[best_flow_idx]:.4f}")
        print(f"Concentration difference: {conc_diffs[best_flow_idx]:.4f}")
        print(f"\nBest solution for concentration (index {best_conc_idx}):")
        print(f"Flow difference: {flow_diffs[best_conc_idx]:.4f}")
        print(f"Concentration difference: {conc_diffs[best_conc_idx]:.4f}")
        
        # Save detailed results
        comparison_data = pd.DataFrame({
            'flow_difference': flow_diffs,
            'conc_difference': conc_diffs,
            'actual_flow': [f.tolist() if isinstance(f, np.ndarray) else f for f in actual_flows],
            'actual_conc': [c.tolist() if isinstance(c, np.ndarray) else c for c in actual_concs]
        })
        comparison_data.to_csv(f"{address}/pareto_comparison_data.csv")
        
        # Save best solutions with their edge lengths
        best_solutions = pd.DataFrame({
            'edge': [f'e{i}' for i in range(1, inlet_edge_idx[-1] + 2)],
            'best_flow_solution': length_values[best_flow_idx],
            'best_conc_solution': length_values[best_conc_idx]
        })
        best_solutions.to_csv(f"{address}/pareto_best_solutions.csv")
        
         # After computing flow_diffs and conc_diffs, add the evaluation:
        print("\nEvaluating Pareto front solutions...")
        evaluations = evaluate_pareto_solutions(flow_diffs, conc_diffs)
    
        # Create comparison visualization
        plt.figure(figsize=(15, 12))
        plt.scatter(flow_diffs, conc_diffs, alpha=0.5, label='All solutions', s=200)
    
        # Plot best solutions from each method
        markers = ['*', '^', 'D', 's', 'P', 'X']
        colors = ['r', 'g', 'b', 'm', 'c', 'y']
    
        for (method_name, eval_data), marker, color in zip(evaluations.items(), markers, colors):
            best_idx = eval_data['best_idx']
            plt.scatter(flow_diffs[best_idx], conc_diffs[best_idx], 
                        marker=marker, color=color, s=800, 
                        label=f'Best ({eval_data["method"]})'
            )
    
        plt.xlabel('Flow Rate Difference')
        plt.ylabel('Concentration Difference')
        plt.title('Best Solutions by Different Metrics')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{address}/pareto_comparison_with_best.png", bbox_inches='tight')
        plt.close()
    
        # Save detailed evaluation results
        evaluation_results = []
        for method_name, eval_data in evaluations.items():
            best_idx = eval_data['best_idx']
            result = {
                'method': eval_data['method'],
                'best_solution_index': best_idx,
                'flow_difference': flow_diffs[best_idx],
                'conc_difference': conc_diffs[best_idx],
                'actual_flow': actual_flows[best_idx].tolist() if isinstance(actual_flows[best_idx], np.ndarray) else actual_flows[best_idx],
                'actual_conc': actual_concs[best_idx].tolist() if isinstance(actual_concs[best_idx], np.ndarray) else actual_concs[best_idx]
            }
            evaluation_results.append(result)
    
        # Save evaluation results
        pd.DataFrame(evaluation_results).to_csv(f"{address}/pareto_evaluation_results.csv")
    
        # Print evaluation summary
        print("\nEvaluation Summary:")
        print("-" * 50)
        for result in evaluation_results:
            print(f"\nMethod: {result['method']}")
            print(f"Best solution index: {result['best_solution_index']}")
            print(f"Flow difference: {result['flow_difference']:.4f}")
            print(f"Concentration difference: {result['conc_difference']:.4f}")
    
        # Save best solutions with their edge lengths
        best_solutions = {}
        for method_name, eval_data in evaluations.items():
            best_idx = eval_data['best_idx']
            best_solutions[f'best_{method_name}'] = length_values[best_idx]
    
        best_solutions_df = pd.DataFrame(best_solutions)
        best_solutions_df.index = [f'e{i}' for i in range(1, inlet_edge_idx[-1] + 2)]
        best_solutions_df.to_csv(f"{address}/pareto_best_solutions_by_method.csv")
    
        return evaluation_results
    
def length_change_func_pareto(
    unks,
    whatToSolve,
    inc_csv,
    args,
    ini_list,
    inlet_node_idx,
    inlet_edge_idx,
    address,
    popSize,
):
    """
    Calculate flow and concentration results for Pareto front solutions.
    Similar to length_change_func but uses pre-loaded length data.

    Args:
        unks (np.ndarray): Array of lengths for evaluation.
        whatToSolve (str or list): Indicates what to solve in the problem.
        inc_csv (str): Filename of the incidence matrix CSV.
        args (tuple): Contains (eta, w, h, n).
        ini_list (list): Initial values of voltage & current sources.
        inlet_node_idx (list): Indices of inlet nodes.
        inlet_edge_idx (list): Indices of inlet edges.
        address (str): Directory path for files.
        popSize (int): Population size for vectorized calculations.

    Returns:
        result (list): Contains two numpy arrays:
            - Scaled outlet flow rates
            - Outlet concentrations (if applicable)
    """
    eta, w, h, n = args
    
    # Setup variables based on what needs to be solved
    (
        isLengthUnknown,
        isSourceUnknown,
        isConcUnknown,
    ) = variable_setup(whatToSolve)

    # Load and prepare incidence matrix
    inc_mat = inc_mat_produce(inc_csv, address)
    inc_mat_vec = np.broadcast_to(inc_mat, (popSize,) + inc_mat.shape)

    # Prepare lists for inlet and outlet
    ini_list_outlet = [i for i in ini_list if i[3] == "o"]
    ini_list_inlet = [i for i in ini_list if i[3] == "i"]

    # Construct block matrix
    arg = (w, h, n)
    A_vec = construct_block_mat_vec(unks, eta, arg, inc_mat_vec, popSize)

    # Construct RHS vector and ground matrices
    b_f = construct_RHS(inc_mat, ini_list)
    b_f_vec = np.broadcast_to(b_f, (popSize,) + b_f.shape)
    
    A_ground_vec, b_f_ground_vec, gnd_idx = ground_block_mat_vec(
        A_vec, inc_mat_vec, b_f_vec, ini_list_outlet
    )

    # Solve Kirchhoff's equations
    i_vec, p_vec, b_f_vec = Kirchhoff_solver_vec(
        A_vec,
        A_ground_vec,
        b_f_vec,
        b_f_ground_vec,
        inc_mat_vec,
        isSourceUnknown,
        ini_list,
        ini_list_outlet,
        ini_list_inlet,
        gnd_idx,
        popSize,
    )

    # Calculate concentrations if required
    conc_vec = None
    if isConcUnknown and len(ini_list_inlet) > 1:
        conc_vec = conc_calculation_vec(
            b_f_vec,
            i_vec,
            inc_mat_vec,
            ini_list_inlet,
            ini_list_outlet,
            inlet_node_idx,
            inlet_edge_idx,
            popSize,
        )

    if conc_vec is not None:
        conc_vec_total = np.concatenate(tuple(conc_vec), axis=1)
    else:
        conc_vec_total = None

    # Calculate scaled outlet flow rates
    i_vec_outlet = i_vec[:, :len(ini_list_outlet)]
    
    # Calculate total flow rate
    current_driven = False
    total_flow_rate = 0
    for i in ini_list_inlet:
        if i[4] == "m^3/s":
            current_driven = True
            total_flow_rate += i[2]
    
    if current_driven:
        total_flow_rate = total_flow_rate * -1 * 10**6 * 60  # Convert to mL/min
    else:
        # For pressure-driven flow, calculate total flow rate from first solution
        A = construct_block_mat(unks[0], eta, arg, inc_mat)
        b_f = construct_RHS(inc_mat, ini_list)
        A_ground, b_f_ground, gnd_idx = ground_block_mat(A, inc_mat, b_f, ini_list_outlet)
        i, p, b_f = Kirchhoff_solver(
            A,
            A_ground,
            b_f,
            b_f_ground,
            inc_mat,
            isSourceUnknown,
            ini_list,
            ini_list_outlet,
            ini_list_inlet,
            gnd_idx,
        )
        inlet_current_vec = i[-len(ini_list_inlet):]
        total_flow_rate = sum([abs(idx) for idx in inlet_current_vec])

    i_vec_outlet_scaled = i_vec_outlet / total_flow_rate
    
    return [i_vec_outlet_scaled, conc_vec_total]


def evaluate_pareto_solutions(flow_diffs, conc_diffs):
    """
    Evaluate Pareto front solutions using multiple metrics.
    
    Args:
        flow_diffs (np.ndarray): Array of flow differences
        conc_diffs (np.ndarray): Array of concentration differences
        
    Returns:
        dict: Dictionary containing evaluation results for each method
    """
    # Normalize the objectives to [0,1] scale
    flow_norm = (flow_diffs - np.min(flow_diffs)) / (np.max(flow_diffs) - np.min(flow_diffs))
    conc_norm = (conc_diffs - np.min(conc_diffs)) / (np.max(conc_diffs) - np.min(conc_diffs))
    
    evaluations = {}
    
    # 1. Weighted Sum Method (try different weights)
    weights = [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3)]  # Different weight combinations
    for w_flow, w_conc in weights:
        weighted_sum = w_flow * flow_norm + w_conc * conc_norm
        evaluations[f'weighted_{w_flow}_{w_conc}'] = {
            'best_idx': np.argmin(weighted_sum),
            'metric': weighted_sum,
            'method': f'Weighted sum (flow:{w_flow}, conc:{w_conc})'
        }
    
    # 2. Trade-off Balance
    trade_off_ratio = flow_norm / (conc_norm + 1e-10)  # Add small number to avoid division by zero
    balanced_idx = np.argmin(np.abs(trade_off_ratio - 1))  # Find most balanced solution
    evaluations['trade_off'] = {
        'best_idx': balanced_idx,
        'metric': np.abs(trade_off_ratio - 1),
        'method': 'Most balanced trade-off'
    }
    
    # 3. Hypervolume Contribution (approximate)
    # Calculate contribution of each point to the dominated hypervolume
    contributions = []
    for i in range(len(flow_norm)):
        # Remove point i and calculate hypervolume
        mask = np.ones(len(flow_norm), dtype=bool)
        mask[i] = False
        remaining_points = np.column_stack((flow_norm[mask], conc_norm[mask]))
        contribution = calculate_hypervolume_contribution(
            np.array([flow_norm[i], conc_norm[i]]), 
            remaining_points
        )
        contributions.append(contribution)
    
    evaluations['hypervolume'] = {
        'best_idx': np.argmax(contributions),
        'metric': contributions,
        'method': 'Maximum hypervolume contribution'
    }
    
    # 4. Euclidean Distance to Ideal Point
    ideal_point = np.array([0, 0])  # Ideal point is origin after normalization
    distances = np.sqrt(flow_norm**2 + conc_norm**2)
    evaluations['euclidean'] = {
        'best_idx': np.argmin(distances),
        'metric': distances,
        'method': 'Minimum distance to ideal point'
    }
    
    return evaluations

def calculate_hypervolume_contribution(point, other_points):
    """
    Calculate the approximate hypervolume contribution of a point.
    
    Args:
        point (np.ndarray): The point to evaluate
        other_points (np.ndarray): Other points in the Pareto front
        
    Returns:
        float: Approximate hypervolume contribution
    """
    # Find the closest dominating and dominated points
    ref_point = np.array([1.1, 1.1])  # Reference point slightly beyond the normalized range
    
    # Calculate area between point and reference point
    area = (ref_point[0] - point[0]) * (ref_point[1] - point[1])
    
    # Subtract areas dominated by other points
    for other in other_points:
        if np.all(other <= point):  # if other point dominates this point
            overlap = (ref_point[0] - other[0]) * (ref_point[1] - other[1])
            area -= overlap
            
    return max(0, area)  # Ensure non-negative contribution


def execute_length_change(
    whatToSolve,
    changing_edges,
    inc_csv,
    length_csv,
    conc_csv,
    args,
    ini_list,
    inlet_node_idx,
    inlet_edge_idx,
    outlet_node_idx,
    outlet_edge_idx,
    conc_wght,
    flow_wght,
    address,
    popSize=200,
    nGen=200,
    *,
    optimizer="nsga2",
    **optimizer_options,
):
    """Optimize channel lengths using NSGA-II, PSO, or multi-start SLSQP.

    The positional parameters and the ``popSize``/``nGen`` defaults preserve
    the original public API. Existing callers therefore continue to use
    NSGA-II unless they explicitly pass ``optimizer="pso"`` or
    ``optimizer="slsqp"``.

    PSO and SLSQP are loaded lazily and receive this module explicitly as their
    circuit solver. This avoids circular imports and keeps the hydraulic and
    concentration calculations identical across all three optimizers.

    By default every backend writes ``new_length_mat.csv`` so the established
    ``electric_analogy_programming.py`` workflow can read the result unchanged.
    Pass ``output_suffix`` when algorithm-specific output files are desired.
    """
    optimizer_key = str(optimizer).strip().lower().replace("_", "").replace("-", "")
    common_args = (
        whatToSolve,
        changing_edges,
        inc_csv,
        length_csv,
        conc_csv,
        args,
        ini_list,
        inlet_node_idx,
        inlet_edge_idx,
        outlet_node_idx,
        outlet_edge_idx,
        conc_wght,
        flow_wght,
        address,
    )

    if optimizer_key in {"nsga2", "nsgaii", "nsga"}:
        if optimizer_options:
            unknown = ", ".join(sorted(optimizer_options))
            raise TypeError(f"Unsupported NSGA-II option(s): {unknown}")
        return _execute_length_change_nsga2(
            *common_args,
            popSize=popSize,
            nGen=nGen,
        )

    solver_module = sys.modules[__name__]
    optimizer_options.setdefault("output_suffix", "")

    if optimizer_key in {"pso", "particleswarm", "particleswarmoptimization"}:
        if __package__:
            from .electric_analogy_pso import execute_length_change as execute_pso
        else:
            from electric_analogy_pso import execute_length_change as execute_pso

        return execute_pso(
            *common_args,
            popSize=popSize,
            nGen=nGen,
            solver_module=solver_module,
            **optimizer_options,
        )

    if optimizer_key in {"slsqp", "multistartslsqp"}:
        if __package__:
            from .electric_analogy_slsqp import execute_length_change as execute_slsqp
        else:
            from electric_analogy_slsqp import execute_length_change as execute_slsqp

        return execute_slsqp(
            *common_args,
            popSize=popSize,
            nGen=nGen,
            solver_module=solver_module,
            **optimizer_options,
        )

    raise ValueError(
        f"Unknown optimizer {optimizer!r}; choose 'nsga2', 'pso', or 'slsqp'."
    )
