import numpy as np
import sympy as sp
import copy
import matplotlib.pyplot as plt
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
LEGACY_OUTPUT_DIR = BASE_DIR / "outputs" / "legacy_symbolic"
LEGACY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% plt rcparams update
plt.rcParams.update(
    {
        "font.size": 50,
        "text.usetex": True,
        "font.family": "times",
        "axes.linewidth": 5,
        "lines.linewidth": 5,
        "xtick.major.width": 5,
        "ytick.major.width": 5,
        "xtick.major.size": 10,
        "ytick.major.size": 10,
        "xtick.major.pad": 10,
        "ytick.major.pad": 10,
        "xtick.minor.width": 5,
        "ytick.minor.width": 5,
        "xtick.minor.size": 10,
        "ytick.minor.size": 10,
        "legend.facecolor": "white",
        "legend.edgecolor": "None",
        "legend.framealpha": 0.5,
    }
)

n = 5
colors = plt.cm.rainbow(np.linspace(0, 1, n))


# %% function define
def rectangular(w, h, n):
    f = lambda n: (1 / (n**5)) * np.tanh(n * np.pi * w / (2 * h))
    g = lambda n: sum([f(i) for i in range(1, n, 2)])

    alpha = 12 * (w / h) * (1 - (h / w) * ((192 / np.pi**5) * g(n)))
    area = w * h
    return alpha, area


def length_to_conc_func(resist, system, resist_var):
    resist_dict = {str(resist_var[i]): resist[i] for i in range(len(resist))}

    LHS = system[0]
    RHS = system[1]

    # Create new equations with substituted resistance values
    # Get all symbols in the equation
    symbols_LHS = LHS.free_symbols
    symbols_RHS = RHS.free_symbols

    # Create substitution dictionary for resistance values
    subs_dict_LHS = {}
    for symbol in symbols_LHS:
        symbol_str = str(symbol)
        # Check if the symbol starts with 'r_' and exists in resist dictionary
        if symbol_str.startswith("r_") and symbol_str in resist_dict:
            subs_dict_LHS[symbol] = resist_dict[symbol_str]

    subs_dict_RHS = {}
    for symbol in symbols_RHS:
        symbol_str = str(symbol)
        # Check if the symbol starts with 'r_' and exists in resist dictionary
        if symbol_str.startswith("r_") and symbol_str in resist_dict:
            subs_dict_RHS[symbol] = resist_dict[symbol_str]

    # Substitute resistance values
    new_LHS = LHS.subs(subs_dict_LHS)
    new_RHS = RHS.subs(subs_dict_RHS)

    # eqn_lambda = [sp.lambdify(tuple(substituted_eqn[i].free_symbols), substituted_eqn[i]) for i in range(len(substituted_eqn))]
    # eqn_soln = [sp.solve(substituted_eqn[i], ([x1_o_1, x1_o_2])) for i in range(len(substituted_eqn))]

    substituted_system = (new_LHS, new_RHS)

    return substituted_system


def create_variable_plots(func, static_grid, variable_grid, resist_var, y_num):
    for num in range(len(static_grid)):
        if num // 2 != 0:
            # Create figure and axes for all plots (5x5 subplots for each variable)
            fig, axes = plt.subplots(8, 7, figsize=(112, 98))

            # Plot for variable a (b and c are static)
            for idx, target_resist in enumerate(resist_var):
                resist_var_exclude = copy.deepcopy(resist_var)
                resist_var_exclude.remove(target_resist)
                for i, resist in enumerate(resist_var_exclude):
                    (row, col) = (idx, i)

                    # For each static value
                    for j, static_val in enumerate(static_grid):
                        y_values = []

                        # For each point in variable grid
                        for var_val in variable_grid:
                            # Create input array with default 0.5 values
                            resist_grid = np.ones(len(resist_var)) * static_grid[num]

                            # Set variable and static values at specified indices
                            resist_grid[idx] = var_val
                            if idx > i:
                                resist_grid[i] = static_val
                            else:
                                resist_grid[i + 1] = static_val

                            # Evaluate function
                            y_values.append(func(resist_grid)[y_num - 1])

                        # Plot
                        axes[row, col].plot(
                            variable_grid,
                            y_values,
                            label=f"{str(resist)}={static_val:e}",
                            color=colors[j],
                        )

                    axes[row, col].set_title(
                        f"x_o_{y_num} vs {str(target_resist)}, varying {str(resist)}"
                    )
                    axes[row, col].grid(True)
                    axes[row, col].legend()

            plt.title(f"x_o_{y_num}_param_{static_grid[num]:e}")
            plt.tight_layout()
            plt.savefig(LEGACY_OUTPUT_DIR / f"x_o_{y_num}_param_{static_grid[num]:e}.png")
            plt.close()


def create_variable_plots_q(func, static_grid, variable_grid, resist_var, y_num):
    for num in range(len(static_grid)):
        if num // 2 != 0:
            # Create figure and axes for all plots (5x5 subplots for each variable)
            fig, axes = plt.subplots(8, 7, figsize=(112, 98))

            # Plot for variable a (b and c are static)
            for idx, target_resist in enumerate(resist_var):
                resist_var_exclude = copy.deepcopy(resist_var)
                resist_var_exclude.remove(target_resist)
                for i, resist in enumerate(resist_var_exclude):
                    (row, col) = (idx, i)

                    # For each static value
                    for j, static_val in enumerate(static_grid):
                        y_values = []

                        # For each point in variable grid
                        for var_val in variable_grid:
                            # Create input array with default 0.5 values
                            resist_grid = np.ones(len(resist_var)) * static_grid[num]

                            # Set variable and static values at specified indices
                            resist_grid[idx] = var_val
                            if idx > i:
                                resist_grid[i] = static_val
                            else:
                                resist_grid[i + 1] = static_val

                            # Evaluate function
                            y_values.append(func(resist_grid)[y_num - 1])

                        # Plot
                        axes[row, col].plot(
                            variable_grid,
                            y_values,
                            label=f"{str(resist)}={static_val:e}",
                            color=colors[j],
                        )

                    axes[row, col].set_title(
                        f"q_o_{y_num} vs {str(target_resist)}, varying {str(resist)}"
                    )
                    axes[row, col].grid(True)
                    axes[row, col].legend()

            plt.title(f"q_o_{y_num}_param_{static_grid[num]:e}")
            plt.tight_layout()
            plt.savefig(LEGACY_OUTPUT_DIR / f"q_o_{y_num}_param_{static_grid[num]:e}.png")
            plt.close()


def printer(variable, filename):
    with open(filename, "w") as f:
        print(sp.latex(variable), file=f)


# %% variable define
address = str(LEGACY_OUTPUT_DIR) + "/"

A = sp.Matrix(
    [
        [1, 0, -1, 0, 0, 0, 0, 0],
        [0, 1, 0, -1, 0, 0, 0, 0],
        [0, 0, 1, 0, -1, 0, 0, 0],
        [0, 0, 1, 0, 0, -1, 0, 0],
        [0, 0, 0, 1, -1, 0, 0, 0],
        [0, 0, 0, 1, 0, -1, 0, 0],
        [0, 0, 0, 0, 1, 0, -1, 0],
        [0, 0, 0, 0, 0, 1, 0, -1],
    ]
)

b = sp.Matrix([0, 0, 0, 0, 0, 0, 0, 0])

f_o_1 = sp.Symbol("f_o_1")
f_o_2 = sp.Symbol("f_o_2")

L = sp.Matrix(
    [
        [20, 0, 0, 0, 0, 0, 0, 0],
        [0, 23, 0, 0, 0, 0, 0, 0],
        [0, 0, 21, 0, 0, 0, 0, 0],
        [0, 0, 0, 59, 0, 0, 0, 0],
        [0, 0, 0, 0, 39, 0, 0, 0],
        [0, 0, 0, 0, 0, 21, 0, 0],
        [0, 0, 0, 0, 0, 0, 20, 0],
        [0, 0, 0, 0, 0, 0, 0, 5],
    ]
) * 10 ** (-3)

# length = [20, 23, 21, 59, 39, 21, 20, 5]
length = [10, 10, 10, 20, 20, 10, 10, 10]
length = [i * 10 ** (-3) for i in length]
length = np.array(length)

w = 500 * 10 ** (-6)
h = 500 * 10 ** (-6)
eta = 1.002 * 10 ** (-3)
n = 20001

alpha, area = rectangular(w, h, n)
R_H_ideal = eta * L / (area**2)
R_original = alpha * R_H_ideal

r_original = alpha * eta * length / (area**2)

q_desired = 1e-6 / 60

f_i_1 = -1 * q_desired
f_i_2 = -1 * q_desired

f = sp.Matrix([f_o_1, f_o_2, 0, 0, 0, 0, f_i_1, f_i_2])

Q_desired = sp.Matrix([q_desired, q_desired])
x1_desired = sp.Matrix([0.7, 0.3])
x2_desired = sp.Matrix([0.3, 0.7])
x1_inlet = sp.Matrix([1, 0])
x2_inlet = sp.Matrix([0, 1])

q_o_1 = sp.Symbol("q_o_1")
q_o_2 = sp.Symbol("q_o_2")
q_m_1 = sp.Symbol("q_m_1")
q_m_2 = sp.Symbol("q_m_2")
q_m_3 = sp.Symbol("q_m_3")
q_m_4 = sp.Symbol("q_m_4")
q_i_1 = sp.Symbol("q_i_1")
q_i_2 = sp.Symbol("q_i_2")

p_o_1 = sp.Symbol("p_o_1")
p_o_2 = sp.Symbol("p_o_2")
p_m_1 = sp.Symbol("p_m_1")
p_m_2 = sp.Symbol("p_m_2")
p_m_3 = sp.Symbol("p_m_3")
p_m_4 = sp.Symbol("p_m_4")
p_i_1 = sp.Symbol("p_i_1")
p_i_2 = sp.Symbol("p_i_2")

x1_o_1 = sp.Symbol("x1_o_1")
x1_o_2 = sp.Symbol("x1_o_2")
x1_m_1 = sp.Symbol("x1_m_1")
x1_m_2 = sp.Symbol("x1_m_2")
x1_m_3 = sp.Symbol("x1_m_3")
x1_m_4 = sp.Symbol("x1_m_4")
x1_i_1 = sp.Symbol("x1_i_1")
x1_i_2 = sp.Symbol("x1_i_2")

x2_o_1 = sp.Symbol("x2_o_1")
x2_o_2 = sp.Symbol("x2_o_2")
x2_m_1 = sp.Symbol("x2_m_1")
x2_m_2 = sp.Symbol("x2_m_2")
x2_m_3 = sp.Symbol("x2_m_3")
x2_m_4 = sp.Symbol("x2_m_4")
x2_i_1 = sp.Symbol("x2_i_1")
x2_i_2 = sp.Symbol("x2_i_2")

r_o_1 = sp.Symbol("r_o_1")
r_o_2 = sp.Symbol("r_o_2")
r_m_1 = sp.Symbol("r_m_1")
r_m_2 = sp.Symbol("r_m_2")
r_m_3 = sp.Symbol("r_m_3")
r_m_4 = sp.Symbol("r_m_4")
r_i_1 = sp.Symbol("r_i_1")
r_i_2 = sp.Symbol("r_i_2")

R = sp.Matrix(
    [
        [r_o_1, 0, 0, 0, 0, 0, 0, 0],
        [0, r_o_2, 0, 0, 0, 0, 0, 0],
        [0, 0, r_m_1, 0, 0, 0, 0, 0],
        [0, 0, 0, r_m_2, 0, 0, 0, 0],
        [0, 0, 0, 0, r_m_3, 0, 0, 0],
        [0, 0, 0, 0, 0, r_m_4, 0, 0],
        [0, 0, 0, 0, 0, 0, r_i_1, 0],
        [0, 0, 0, 0, 0, 0, 0, r_i_2],
    ]
)

# R = R_original

# %% flow rate computation - matrix equation define
const_col = sp.zeros(16, 2)
const_col[16] = 1
const_col[19] = 1
const_col

LHS = R.row_join(A).col_join(A.T.row_join(sp.zeros(8, 8)))

LHS = LHS.row_join(-1 * const_col)

const_row = const_col.T.row_join(sp.zeros(2, 2))

LHS = LHS.col_join(const_row)

f_modified = f
f_modified[0] = 0
f_modified[1] = 0

RHS = b.col_join(f_modified).col_join(sp.Matrix([0, 0]))

q = sp.Matrix([q_o_1, q_o_2, q_m_1, q_m_2, q_m_3, q_m_4, q_i_1, q_i_2])
p = sp.Matrix([p_o_1, p_o_2, p_m_1, p_m_2, p_m_3, p_m_4, p_i_1, p_i_2])
q_o = sp.Matrix([q_o_1, q_o_2])
q_m = sp.Matrix([q_m_1, q_m_2, q_m_3, q_m_4])
q_i = sp.Matrix([q_i_1, q_i_2])

# %% flow rate computation
q_p_computed = sp.linsolve((LHS, RHS))

q_p_computed = q_p_computed.args[0]
printer(q_p_computed, f"{address}q_p_computed.txt")

q_computed = sp.zeros(8, 1)
for idx in range(8):
    q_computed[idx] = q_p_computed[idx]

q_compute = sp.simplify(q_computed)

# q_computed[gnd_idx] = Q_desired[gnd_idx]

printer(q_computed, f"{address}q_computed.txt")

q_o_computed = copy.deepcopy(q_computed)
for idx in range(7, 1, -1):
    q_o_computed.row_del(idx)

q_m_computed = copy.deepcopy(q_computed)
for idx in range(7, 5, -1):
    q_m_computed.row_del(idx)
q_m_computed.row_del(0)
q_m_computed.row_del(0)

q_i_computed = copy.deepcopy(q_computed)
for idx in range(5, -1, -1):
    q_i_computed.row_del(0)

printer(q_o_computed, f"{address}q_o_computed.txt")
printer(q_m_computed, f"{address}q_m_computed.txt")
printer(q_i_computed, f"{address}q_i_computed.txt")

Q_o_computed = sp.diag(q_o_computed[0], q_o_computed[1])
Q_m_computed = sp.diag(
    q_m_computed[0], q_m_computed[1], q_m_computed[2], q_m_computed[3]
)
Q_i_computed = sp.diag(q_i_computed[0], q_i_computed[1])

Q_desired_diag = sp.diag(Q_desired[0], Q_desired[1])

F_i = sp.diag(f_i_1, f_i_2)

# %% concentration computation - matrix equation define
# Handle nodes with multiple outlets
two_or_more_outlet_list = []
inc_mat = np.array(A)
for idx in range(8):
    if list(inc_mat.T[idx]).count(-1) > 1:
        two_or_more_outlet_list.append(idx)

elem_list = []
for idx in two_or_more_outlet_list:
    pvt = np.zeros(8)
    for j in range(8):
        if inc_mat.T[idx][j] == 1:
            pvt[j] = 1
    for j in range(8):
        if inc_mat.T[idx][j] == -1:
            tmp = pvt.copy()
            tmp[j] = -1
            elem_list.append(tmp)

two_or_more_outlet_mat = np.vstack(tuple(elem_list)) if elem_list else None
if two_or_more_outlet_mat is not None:
    two_or_more_outlet_mats = np.hsplit(two_or_more_outlet_mat, [6])

two_or_more_outlet_mats_LHS = np.hsplit(two_or_more_outlet_mats[0], [2])

E_o = sp.Matrix(two_or_more_outlet_mats_LHS[0])
E_m = sp.Matrix(two_or_more_outlet_mats_LHS[1])
E_i = sp.Matrix(two_or_more_outlet_mats[1])

A_oo = A[:2, :2]
A_om = A[:2, 2:6]
A_oi = A[:2, 6:8]
A_mo = A[2:6, :2]
A_mm = A[2:6, 2:6]
A_mi = A[2:6, 6:8]
A_io = A[6:8, :2]
A_im = A[6:8, 2:6]
A_ii = A[6:8, 6:8]

A_T_oo = A_oo.T
A_T_om = A_om.T
A_T_oi = A_oi.T
A_T_mo = A_mo.T
A_T_mm = A_mm.T
A_T_mi = A_mi.T
A_T_io = A_io.T
A_T_im = A_im.T
A_T_ii = A_ii.T

x1_i_1 = x1_inlet[0]
x1_i_2 = x1_inlet[1]

x2_i_1 = x2_inlet[0]
x2_i_2 = x2_inlet[1]

x1_o = sp.Matrix([x1_o_1, x1_o_2])
x1_m = sp.Matrix([x1_m_1, x1_m_2, x1_m_3, x1_m_4])
x1_i = sp.Matrix([x1_i_1, x1_i_2])

x2_o = sp.Matrix([x2_o_1, x2_o_2])
x2_m = sp.Matrix([x2_m_1, x2_m_2, x2_m_3, x2_m_4])
x2_i = sp.Matrix([x2_i_1, x2_i_2])

LHS_conc = (
    (A_T_oo * Q_o_computed - Q_desired_diag)
    .row_join(A_T_mo * Q_m_computed)
    .col_join((A_T_om * Q_o_computed).row_join(A_T_mm * Q_m_computed))
    .col_join((A_T_oi * Q_o_computed).row_join(A_T_mi * Q_m_computed))
    .col_join(E_o.row_join(E_m))
)

RHS_conc1 = (
    (-1 * A_T_io * Q_i_computed * x1_i)
    .col_join(-1 * A_T_im * Q_i_computed * x1_i)
    .col_join(((F_i - A_T_ii * Q_i_computed) * x1_i))
    .col_join(-1 * E_i * x1_i)
)

RHS_conc2 = (
    (-1 * A_T_io * Q_i_computed * x2_i)
    .col_join(-1 * A_T_im * Q_i_computed * x2_i)
    .col_join(((F_i - A_T_ii * Q_i_computed) * x2_i))
    .col_join(-1 * E_i * x2_i)
)

printer(LHS_conc, f"{address}LHS_conc.txt")
printer(RHS_conc1, f"{address}RHS_conc1.txt")
printer(RHS_conc2, f"{address}RHS_conc2.txt")

LHS_normal = LHS_conc.T * LHS_conc
LHS_normal = sp.simplify(LHS_normal)

RHS_normal1 = LHS_conc.T * RHS_conc1
RHS_normal2 = LHS_conc.T * RHS_conc2

printer(LHS_normal, f"{address}LHS_normal.txt")
printer(RHS_normal1, f"{address}RHS_normal1.txt")
printer(RHS_normal2, f"{address}RHS_normal2.txt")

# %% solving system - variable define
unknowns1 = sp.Matrix([x1_o_1, x1_o_2, x1_m_1, x1_m_2, x1_m_3, x1_m_4])
unknowns2 = sp.Matrix([x2_o_1, x2_o_2, x2_m_1, x2_m_2, x2_m_3, x2_m_4])

system1 = (LHS_normal, RHS_normal1)
system2 = (LHS_normal, RHS_normal2)

var1 = [x1_o_1, x1_o_2, x1_m_1, x1_m_2, x1_m_3, x1_m_4]
var2 = [x2_o_1, x2_o_2, x2_m_1, x2_m_2, x2_m_3, x2_m_4]

resist_var = [r_o_1, r_o_2, r_m_1, r_m_2, r_m_3, r_m_4, r_i_1, r_i_2]

static_grid = np.linspace(5e6, 1e11, 5)
variable_grid = np.linspace(5e6, 1e11, 100)

func1 = (
    lambda x: np.array(
        sp.Matrix(
            sp.linsolve(length_to_conc_func(x, system1, resist_var), var1).args[0][0:2]
        )
        - x1_desired
    )
    ** 2
)
func2 = (
    lambda x: np.array(
        sp.Matrix(
            sp.linsolve(length_to_conc_func(x, system2, resist_var), var2).args[0][0:2]
        )
        - x2_desired
    )
    ** 2
)
func_q_o = (
    lambda x: np.array(
        length_to_conc_func(x, (q_o_computed, sp.zeros(2, 1)), resist_var)[0]
        - Q_desired
    )
    ** 2
)

# %% flow rate objective function computation
q_o_diff = q_o_computed - Q_desired
q_objective_1 = q_o_diff[0] ** 2
q_objective_2 = q_o_diff[1] ** 2

q_objective_1 = sp.simplify(q_objective_1)
q_objective_2 = sp.simplify(q_objective_2)

printer(q_objective_1, f"{address}q_objective_1.txt")
printer(q_objective_2, f"{address}q_objective_2.txt")

# 1st derivatives, 1st output
q_objective_1_r_o_1 = sp.diff(q_objective_1, r_o_1)
q_objective_1_r_o_2 = sp.diff(q_objective_1, r_o_2)
q_objective_1_r_m_1 = sp.diff(q_objective_1, r_m_1)
q_objective_1_r_m_2 = sp.diff(q_objective_1, r_m_2)
q_objective_1_r_m_3 = sp.diff(q_objective_1, r_m_3)
q_objective_1_r_m_4 = sp.diff(q_objective_1, r_m_4)
q_objective_1_r_i_1 = sp.diff(q_objective_1, r_i_1)
q_objective_1_r_i_2 = sp.diff(q_objective_1, r_i_2)

printer(q_objective_1_r_o_1, f"{address}q_objective_1_r_o_1.txt")
printer(q_objective_1_r_o_2, f"{address}q_objective_1_r_o_2.txt")
printer(q_objective_1_r_m_1, f"{address}q_objective_1_r_m_1.txt")
printer(q_objective_1_r_m_2, f"{address}q_objective_1_r_m_2.txt")
printer(q_objective_1_r_m_3, f"{address}q_objective_1_r_m_3.txt")
printer(q_objective_1_r_m_4, f"{address}q_objective_1_r_m_4.txt")
printer(q_objective_1_r_i_1, f"{address}q_objective_1_r_i_1.txt")
printer(q_objective_1_r_i_2, f"{address}q_objective_1_r_i_2.txt")

# 2nd derivatives, 1st output
q_objective_1_r_o_1_r_o_1 = sp.simplify(sp.diff(q_objective_1_r_o_1, r_o_1))
q_objective_1_r_o_2_r_o_2 = sp.simplify(sp.diff(q_objective_1_r_o_2, r_o_2))
q_objective_1_r_m_1_r_m_1 = sp.simplify(sp.diff(q_objective_1_r_m_1, r_m_1))
q_objective_1_r_m_2_r_m_2 = sp.simplify(sp.diff(q_objective_1_r_m_2, r_m_2))
q_objective_1_r_m_3_r_m_3 = sp.simplify(sp.diff(q_objective_1_r_m_3, r_m_3))
q_objective_1_r_m_4_r_m_4 = sp.simplify(sp.diff(q_objective_1_r_m_4, r_m_4))
q_objective_1_r_i_1_r_i_1 = sp.simplify(sp.diff(q_objective_1_r_i_1, r_i_1))
q_objective_1_r_i_2_r_i_2 = sp.simplify(sp.diff(q_objective_1_r_i_2, r_i_2))

printer(q_objective_1_r_o_1_r_o_1, f"{address}q_objective_1_r_o_1_r_o_1.txt")
printer(q_objective_1_r_o_2_r_o_2, f"{address}q_objective_1_r_o_2_r_o_2.txt")
printer(q_objective_1_r_m_1_r_m_1, f"{address}q_objective_1_r_m_1_r_m_1.txt")
printer(q_objective_1_r_m_2_r_m_2, f"{address}q_objective_1_r_m_2_r_m_2.txt")
printer(q_objective_1_r_m_3_r_m_3, f"{address}q_objective_1_r_m_3_r_m_3.txt")
printer(q_objective_1_r_m_4_r_m_4, f"{address}q_objective_1_r_m_4_r_m_4.txt")
printer(q_objective_1_r_i_1_r_i_1, f"{address}q_objective_1_r_i_1_r_i_1.txt")
printer(q_objective_1_r_i_2_r_i_2, f"{address}q_objective_1_r_i_2_r_i_2.txt")

# 1st derivatives, 2nd output
q_objective_2_r_o_1 = sp.diff(q_objective_2, r_o_1)
q_objective_2_r_o_2 = sp.diff(q_objective_2, r_o_2)
q_objective_2_r_m_1 = sp.diff(q_objective_2, r_m_1)
q_objective_2_r_m_2 = sp.diff(q_objective_2, r_m_2)
q_objective_2_r_m_3 = sp.diff(q_objective_2, r_m_3)
q_objective_2_r_m_4 = sp.diff(q_objective_2, r_m_4)
q_objective_2_r_i_1 = sp.diff(q_objective_2, r_i_1)
q_objective_2_r_i_2 = sp.diff(q_objective_2, r_i_2)

printer(q_objective_2_r_o_1, f"{address}q_objective_2_r_o_1.txt")
printer(q_objective_2_r_o_2, f"{address}q_objective_2_r_o_2.txt")
printer(q_objective_2_r_m_1, f"{address}q_objective_2_r_m_1.txt")
printer(q_objective_2_r_m_2, f"{address}q_objective_2_r_m_2.txt")
printer(q_objective_2_r_m_3, f"{address}q_objective_2_r_m_3.txt")
printer(q_objective_2_r_m_4, f"{address}q_objective_2_r_m_4.txt")
printer(q_objective_2_r_i_1, f"{address}q_objective_2_r_i_1.txt")
printer(q_objective_2_r_i_2, f"{address}q_objective_2_r_i_2.txt")

# 2nd derivatives, 2nd output
q_objective_2_r_o_1_r_o_1 = sp.simplify(sp.diff(q_objective_2_r_o_1, r_o_1))
q_objective_2_r_o_2_r_o_2 = sp.simplify(sp.diff(q_objective_2_r_o_2, r_o_2))
q_objective_2_r_m_1_r_m_1 = sp.simplify(sp.diff(q_objective_2_r_m_1, r_m_1))
q_objective_2_r_m_2_r_m_2 = sp.simplify(sp.diff(q_objective_2_r_m_2, r_m_2))
q_objective_2_r_m_3_r_m_3 = sp.simplify(sp.diff(q_objective_2_r_m_3, r_m_3))
q_objective_2_r_m_4_r_m_4 = sp.simplify(sp.diff(q_objective_2_r_m_4, r_m_4))
q_objective_2_r_i_1_r_i_1 = sp.simplify(sp.diff(q_objective_2_r_i_1, r_i_1))
q_objective_2_r_i_2_r_i_2 = sp.simplify(sp.diff(q_objective_2_r_i_2, r_i_2))

printer(q_objective_2_r_o_1_r_o_1, f"{address}q_objective_2_r_o_1_r_o_1.txt")
printer(q_objective_2_r_o_2_r_o_2, f"{address}q_objective_2_r_o_2_r_o_2.txt")
printer(q_objective_2_r_m_1_r_m_1, f"{address}q_objective_2_r_m_1_r_m_1.txt")
printer(q_objective_2_r_m_2_r_m_2, f"{address}q_objective_2_r_m_2_r_m_2.txt")
printer(q_objective_2_r_m_3_r_m_3, f"{address}q_objective_2_r_m_3_r_m_3.txt")
printer(q_objective_2_r_m_4_r_m_4, f"{address}q_objective_2_r_m_4_r_m_4.txt")
printer(q_objective_2_r_i_1_r_i_1, f"{address}q_objective_2_r_i_1_r_i_1.txt")
printer(q_objective_2_r_i_2_r_i_2, f"{address}q_objective_2_r_i_2_r_i_2.txt")

# acquire critical points of 1st derivative
q_objective_1_r_o_1_solve = sp.solve(q_objective_1_r_o_1, r_o_1)
q_objective_1_r_o_2_solve = sp.solve(q_objective_1_r_o_2, r_o_2)
q_objective_1_r_m_1_solve = sp.solve(q_objective_1_r_m_1, r_m_1)
q_objective_1_r_m_2_solve = sp.solve(q_objective_1_r_m_2, r_m_2)
q_objective_1_r_m_3_solve = sp.solve(q_objective_1_r_m_3, r_m_3)
q_objective_1_r_m_4_solve = sp.solve(q_objective_1_r_m_4, r_m_4)
q_objective_1_r_i_1_solve = sp.solve(q_objective_1_r_i_1, r_i_1)
q_objective_1_r_i_2_solve = sp.solve(q_objective_1_r_i_2, r_i_2)

q_objective_2_r_o_1_solve = sp.solve(q_objective_2_r_o_1, r_o_1)
q_objective_2_r_o_2_solve = sp.solve(q_objective_2_r_o_2, r_o_2)
q_objective_2_r_m_1_solve = sp.solve(q_objective_2_r_m_1, r_m_1)
q_objective_2_r_m_2_solve = sp.solve(q_objective_2_r_m_2, r_m_2)
q_objective_2_r_m_3_solve = sp.solve(q_objective_2_r_m_3, r_m_3)
q_objective_2_r_m_4_solve = sp.solve(q_objective_2_r_m_4, r_m_4)
q_objective_2_r_i_1_solve = sp.solve(q_objective_2_r_i_1, r_i_1)
q_objective_2_r_i_2_solve = sp.solve(q_objective_2_r_i_2, r_i_2)

q_objective_1_r_o_1_solve = q_objective_1_r_o_1_solve[0]
q_objective_1_r_o_2_solve = q_objective_1_r_o_2_solve[0]
q_objective_1_r_m_1_solve = q_objective_1_r_m_1_solve[0]
q_objective_1_r_m_2_solve = q_objective_1_r_m_2_solve[0]
q_objective_1_r_m_3_solve = q_objective_1_r_m_3_solve[0]
q_objective_1_r_m_4_solve = q_objective_1_r_m_4_solve[0]

q_objective_2_r_o_1_solve = q_objective_2_r_o_1_solve[0]
q_objective_2_r_o_2_solve = q_objective_2_r_o_2_solve[0]
q_objective_2_r_m_1_solve = q_objective_2_r_m_1_solve[0]
q_objective_2_r_m_2_solve = q_objective_2_r_m_2_solve[0]
q_objective_2_r_m_3_solve = q_objective_2_r_m_3_solve[0]
q_objective_2_r_m_4_solve = q_objective_2_r_m_4_solve[0]

printer(q_objective_1_r_o_1_solve, f"{address}q_objective_1_r_o_1_solve.txt")
printer(q_objective_1_r_o_2_solve, f"{address}q_objective_1_r_o_2_solve.txt")
printer(q_objective_1_r_m_1_solve, f"{address}q_objective_1_r_m_1_solve.txt")
printer(q_objective_1_r_m_2_solve, f"{address}q_objective_1_r_m_2_solve.txt")
printer(q_objective_1_r_m_3_solve, f"{address}q_objective_1_r_m_3_solve.txt")
printer(q_objective_1_r_m_4_solve, f"{address}q_objective_1_r_m_4_solve.txt")
printer(q_objective_1_r_i_1_solve, f"{address}q_objective_1_r_i_1_solve.txt")
printer(q_objective_1_r_i_2_solve, f"{address}q_objective_1_r_i_2_solve.txt")

printer(q_objective_2_r_o_1_solve, f"{address}q_objective_2_r_o_1_solve.txt")
printer(q_objective_2_r_o_2_solve, f"{address}q_objective_2_r_o_2_solve.txt")
printer(q_objective_2_r_m_1_solve, f"{address}q_objective_2_r_m_1_solve.txt")
printer(q_objective_2_r_m_2_solve, f"{address}q_objective_2_r_m_2_solve.txt")
printer(q_objective_2_r_m_3_solve, f"{address}q_objective_2_r_m_3_solve.txt")
printer(q_objective_2_r_m_4_solve, f"{address}q_objective_2_r_m_4_solve.txt")
printer(q_objective_2_r_i_1_solve, f"{address}q_objective_2_r_i_1_solve.txt")
printer(q_objective_2_r_i_2_solve, f"{address}q_objective_2_r_i_2_solve.txt")

# acquire critical points of 2nd derivative
q_objective_1_r_o_1_r_o_1_solve = sp.solve(q_objective_1_r_o_1_r_o_1, r_o_1)
q_objective_1_r_o_2_r_o_2_solve = sp.solve(q_objective_1_r_o_2_r_o_2, r_o_2)
q_objective_1_r_m_1_r_m_1_solve = sp.solve(q_objective_1_r_m_1_r_m_1, r_m_1)
q_objective_1_r_m_2_r_m_2_solve = sp.solve(q_objective_1_r_m_2_r_m_2, r_m_2)
q_objective_1_r_m_3_r_m_3_solve = sp.solve(q_objective_1_r_m_3_r_m_3, r_m_3)
q_objective_1_r_m_4_r_m_4_solve = sp.solve(q_objective_1_r_m_4_r_m_4, r_m_4)
q_objective_1_r_i_1_r_i_1_solve = sp.solve(q_objective_1_r_i_1_r_i_1, r_i_1)
q_objective_1_r_i_2_r_i_2_solve = sp.solve(q_objective_1_r_i_2_r_i_2, r_i_2)

q_objective_2_r_o_1_r_o_1_solve = sp.solve(q_objective_2_r_o_1_r_o_1, r_o_1)
q_objective_2_r_o_2_r_o_2_solve = sp.solve(q_objective_2_r_o_2_r_o_2, r_o_2)
q_objective_2_r_m_1_r_m_1_solve = sp.solve(q_objective_2_r_m_1_r_m_1, r_m_1)
q_objective_2_r_m_2_r_m_2_solve = sp.solve(q_objective_2_r_m_2_r_m_2, r_m_2)
q_objective_2_r_m_3_r_m_3_solve = sp.solve(q_objective_2_r_m_3_r_m_3, r_m_3)
q_objective_2_r_m_4_r_m_4_solve = sp.solve(q_objective_2_r_m_4_r_m_4, r_m_4)
q_objective_2_r_i_1_r_i_1_solve = sp.solve(q_objective_2_r_i_1_r_i_1, r_i_1)
q_objective_2_r_i_2_r_i_2_solve = sp.solve(q_objective_2_r_i_2_r_i_2, r_i_2)

q_objective_1_r_o_1_r_o_1_solve_1 = q_objective_1_r_o_1_r_o_1_solve[0]
q_objective_1_r_o_2_r_o_2_solve_1 = q_objective_1_r_o_2_r_o_2_solve[0]
q_objective_1_r_m_1_r_m_1_solve_1 = q_objective_1_r_m_1_r_m_1_solve[0]
q_objective_1_r_m_2_r_m_2_solve_1 = q_objective_1_r_m_2_r_m_2_solve[0]
q_objective_1_r_m_3_r_m_3_solve_1 = q_objective_1_r_m_3_r_m_3_solve[0]
q_objective_1_r_m_4_r_m_4_solve_1 = q_objective_1_r_m_4_r_m_4_solve[0]

q_objective_1_r_o_1_r_o_1_solve_2 = q_objective_1_r_o_1_r_o_1_solve[1]
q_objective_1_r_o_2_r_o_2_solve_2 = q_objective_1_r_o_2_r_o_2_solve[1]
q_objective_1_r_m_1_r_m_1_solve_2 = q_objective_1_r_m_1_r_m_1_solve[1]
q_objective_1_r_m_2_r_m_2_solve_2 = q_objective_1_r_m_2_r_m_2_solve[1]
q_objective_1_r_m_3_r_m_3_solve_2 = q_objective_1_r_m_3_r_m_3_solve[1]
q_objective_1_r_m_4_r_m_4_solve_2 = q_objective_1_r_m_4_r_m_4_solve[1]

q_objective_2_r_o_1_r_o_1_solve_1 = q_objective_2_r_o_1_r_o_1_solve[0]
q_objective_2_r_o_2_r_o_2_solve_1 = q_objective_2_r_o_2_r_o_2_solve[0]
q_objective_2_r_m_1_r_m_1_solve_1 = q_objective_2_r_m_1_r_m_1_solve[0]
q_objective_2_r_m_2_r_m_2_solve_1 = q_objective_2_r_m_2_r_m_2_solve[0]
q_objective_2_r_m_3_r_m_3_solve_1 = q_objective_2_r_m_3_r_m_3_solve[0]
q_objective_2_r_m_4_r_m_4_solve_1 = q_objective_2_r_m_4_r_m_4_solve[0]

q_objective_2_r_o_1_r_o_1_solve_2 = q_objective_2_r_o_1_r_o_1_solve[1]
q_objective_2_r_o_2_r_o_2_solve_2 = q_objective_2_r_o_2_r_o_2_solve[1]
q_objective_2_r_m_1_r_m_1_solve_2 = q_objective_2_r_m_1_r_m_1_solve[1]
q_objective_2_r_m_2_r_m_2_solve_2 = q_objective_2_r_m_2_r_m_2_solve[1]
q_objective_2_r_m_3_r_m_3_solve_2 = q_objective_2_r_m_3_r_m_3_solve[1]
q_objective_2_r_m_4_r_m_4_solve_2 = q_objective_2_r_m_4_r_m_4_solve[1]

printer(
    q_objective_1_r_o_1_r_o_1_solve_1, f"{address}q_objective_1_r_o_1_r_o_1_solve_1.txt"
)
printer(
    q_objective_1_r_o_2_r_o_2_solve_1, f"{address}q_objective_1_r_o_2_r_o_2_solve_1.txt"
)
printer(
    q_objective_1_r_m_1_r_m_1_solve_1, f"{address}q_objective_1_r_m_1_r_m_1_solve_1.txt"
)
printer(
    q_objective_1_r_m_2_r_m_2_solve_1, f"{address}q_objective_1_r_m_2_r_m_2_solve_1.txt"
)
printer(
    q_objective_1_r_m_3_r_m_3_solve_1, f"{address}q_objective_1_r_m_3_r_m_3_solve_1.txt"
)
printer(
    q_objective_1_r_m_4_r_m_4_solve_1, f"{address}q_objective_1_r_m_4_r_m_4_solve_1.txt"
)
printer(
    q_objective_1_r_i_1_r_i_1_solve, f"{address}q_objective_1_r_i_1_r_i_1_solve.txt"
)
printer(
    q_objective_1_r_i_2_r_i_2_solve, f"{address}q_objective_1_r_i_2_r_i_2_solve.txt"
)

printer(
    q_objective_2_r_o_1_r_o_1_solve_1, f"{address}q_objective_2_r_o_1_r_o_1_solve_1.txt"
)
printer(
    q_objective_2_r_o_2_r_o_2_solve_1, f"{address}q_objective_2_r_o_2_r_o_2_solve_1.txt"
)
printer(
    q_objective_2_r_m_1_r_m_1_solve_1, f"{address}q_objective_2_r_m_1_r_m_1_solve_1.txt"
)
printer(
    q_objective_2_r_m_2_r_m_2_solve_1, f"{address}q_objective_2_r_m_2_r_m_2_solve_1.txt"
)
printer(
    q_objective_2_r_m_3_r_m_3_solve_1, f"{address}q_objective_2_r_m_3_r_m_3_solve_1.txt"
)
printer(
    q_objective_2_r_m_4_r_m_4_solve_1, f"{address}q_objective_2_r_m_4_r_m_4_solve_1.txt"
)
printer(
    q_objective_2_r_i_1_r_i_1_solve, f"{address}q_objective_2_r_i_1_r_i_1_solve.txt"
)
printer(
    q_objective_2_r_i_2_r_i_2_solve, f"{address}q_objective_2_r_i_2_r_i_2_solve.txt"
)

printer(
    q_objective_1_r_o_1_r_o_1_solve_2, f"{address}q_objective_1_r_o_1_r_o_1_solve_2.txt"
)
printer(
    q_objective_1_r_o_2_r_o_2_solve_2, f"{address}q_objective_1_r_o_2_r_o_2_solve_2.txt"
)
printer(
    q_objective_1_r_m_1_r_m_1_solve_2, f"{address}q_objective_1_r_m_1_r_m_1_solve_2.txt"
)
printer(
    q_objective_1_r_m_2_r_m_2_solve_2, f"{address}q_objective_1_r_m_2_r_m_2_solve_2.txt"
)
printer(
    q_objective_1_r_m_3_r_m_3_solve_2, f"{address}q_objective_1_r_m_3_r_m_3_solve_2.txt"
)
printer(
    q_objective_1_r_m_4_r_m_4_solve_2, f"{address}q_objective_1_r_m_4_r_m_4_solve_2.txt"
)

printer(
    q_objective_2_r_o_1_r_o_1_solve_2, f"{address}q_objective_2_r_o_1_r_o_1_solve_2.txt"
)
printer(
    q_objective_2_r_o_2_r_o_2_solve_2, f"{address}q_objective_2_r_o_2_r_o_2_solve_2.txt"
)
printer(
    q_objective_2_r_m_1_r_m_1_solve_2, f"{address}q_objective_2_r_m_1_r_m_1_solve_2.txt"
)
printer(
    q_objective_2_r_m_2_r_m_2_solve_2, f"{address}q_objective_2_r_m_2_r_m_2_solve_2.txt"
)
printer(
    q_objective_2_r_m_3_r_m_3_solve_2, f"{address}q_objective_2_r_m_3_r_m_3_solve_2.txt"
)
printer(
    q_objective_2_r_m_4_r_m_4_solve_2, f"{address}q_objective_2_r_m_4_r_m_4_solve_2.txt"
)

# %% create plots between objectives and each variables
create_variable_plots_q(func_q_o, static_grid, variable_grid, resist_var, 1)
create_variable_plots_q(func_q_o, static_grid, variable_grid, resist_var, 2)
create_variable_plots(func1, static_grid, variable_grid, resist_var, 1)
create_variable_plots(func2, static_grid, variable_grid, resist_var, 2)

# %% read convexity map of the plots and create heatmap
# manually indicate the convexity of each plots created from above, and save them as CV_NCV_map.xlsx
from matplotlib.patches import Patch
import pandas as pd

# Load all sheets into a dict of DataFrames
path = BASE_DIR / "CV_NCV_map.xlsx"
sheets = pd.read_excel(path, sheet_name=None, index_col=0)

# Determine dimensions
sheet_names = list(sheets.keys())
nlayers = len(sheet_names)
first_df = sheets[sheet_names[0]]
nx, ny = first_df.shape

# Build a 3D array of categories
class_cube = np.empty((nx, ny, nlayers), dtype=object)
for k, name in enumerate(sheet_names):
    df = sheets[name]
    if df.shape != (nx, ny):
        raise ValueError(f"Sheet '{name}' has shape {df.shape}, expected {(nx,ny)}")
    class_cube[:, :, k] = df.values

# Boolean mask: show every voxel
filled = np.ones_like(class_cube, dtype=bool)

# Identify all unique categories
unique_vals = pd.unique(class_cube.ravel())
n = len(unique_vals)

# Separate real categories vs NaN
real_vals = [v for v in unique_vals if not pd.isna(v)]
has_nan = any(pd.isna(unique_vals))

# Grab a rainbow of N colors, then set alpha=0.4
colors = plt.cm.rainbow(np.linspace(0, 1, n))  # Nx4 RGBA
colors[:, 3] = 0.4  # uniform transparency

# Build dictionary of RGBA values
rgba_map = dict(zip(unique_vals, colors))

# Build facecolors as before
facecolors = np.zeros((nx, ny, nlayers, 4), dtype=float)
for val, rgba in rgba_map.items():
    facecolors[class_cube == val] = rgba

# Plot with voxels
fig = plt.figure(figsize=(25, 25))
ax = fig.add_subplot(111, projection="3d")
ax.voxels(filled, facecolors=facecolors, edgecolor="k", linewidth=0.2)

# Set the view
ax.view_init(elev=30, azim=210)

# Fix the data‑limits so they go from 0 to (n‑1) on each axis
ax.set_xlim(0, nx - 1)
ax.set_ylim(0, ny - 1)
ax.set_zlim(0, nlayers - 1)

# Give the box an aspect ratio matching tick counts
ax.set_box_aspect((nx + 2, ny + 2, nlayers))

# 9. Label axes and ticks
# ax.set_xlabel('Row')
# ax.set_ylabel('Column')
# ax.set_zlabel('Layer')

labels = [
    r"$r_{\mathrm{o1}}$",
    r"$r_{\mathrm{o2}}$",
    r"$r_{\mathrm{m1}}$",
    r"$r_{\mathrm{m2}}$",
    r"$r_{\mathrm{m3}}$",
    r"$r_{\mathrm{m4}}$",
    r"$r_{\mathrm{i1}}$",
    r"$r_{\mathrm{i2}}$",
]
n = len(labels)

sheet_label = [
    r"$x_{\mathrm{o1, 1}}$",
    r"$x_{\mathrm{o1, 2}}$",
    r"$x_{\mathrm{o1, 3}}$",
    r"$x_{\mathrm{o2, 1}}$",
    r"$x_{\mathrm{o2, 2}}$",
    r"$x_{\mathrm{o2, 3}}$",
    r"$q_{\mathrm{o1, 1}}$",
    r"$q_{\mathrm{o1, 2}}$",
    r"$q_{\mathrm{o1, 3}}$",
    r"$q_{\mathrm{o2, 1}}$",
    r"$q_{\mathrm{o2, 2}}$",
    r"$q_{\mathrm{o2, 3}}$",
]

ax.set_xticks(np.arange(n))
ax.set_xticklabels(labels, ha="left")
ax.set_yticks(np.arange(n))
ax.set_yticklabels(labels, ha="right")
ax.set_zticks(np.arange(nlayers))
ax.set_zticklabels(sheet_label, ha="right")
ax.tick_params(axis="z", pad=15)

# In the legend, add the NaN patch if needed:
legend_handles = [
    Patch(facecolor=rgba_map[v], edgecolor="k", label=str(v)) for v in real_vals
]
if has_nan:
    legend_handles.append(Patch(facecolor=(1, 1, 1, 1), edgecolor="k", label="NaN"))

leg = ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.05, 1))

note = r"$\textrm{subscript 1} = 5.0\times10^{10} \\ \textrm{subscript 2} = 7.5\times10^{10} \\ \textrm{subscript 3} = 1.0\times10^{11}$"
# Force a draw so we can query its exact size/position
fig.canvas.draw()
# Get the legend’s bounding box in display coords
bbox_disp = leg.get_window_extent()
# Convert that to figure coords (0..1)
bbox_fig = bbox_disp.transformed(fig.transFigure.inverted())
# Place notes just to the right of that box:
x_text = bbox_fig.x0 - bbox_fig.width
y_text = bbox_fig.y0 - bbox_fig.height / 2 - 0.02
# Add the text
fig.text(
    x_text, y_text, note, fontsize=50, va="bottom", ha="left", multialignment="left"
)

plt.tight_layout()
plt.show()
