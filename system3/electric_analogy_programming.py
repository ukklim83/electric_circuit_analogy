from electric_analogy import *


def main():
    address = "./system3"

    whatToSolve = [
        "length",
        "source",
        "concentration",
        # "optimization",
        # "compare_optimization",
    ]  # str: choose what to solve; 'length', 'source', 'concentration' or 'optimization'
    inc_csv = "incidence_mat.csv"  # str: incidence_matrix filename
    length_csv = "length_mat.csv"  # str: length_matrix filename
    conc_csv = "concentration_mat.csv"  # str: concentration_matrix filename

    ############################################
    # <Rule for incidence matrix>              #
    # node indices: "outlet first, inlet last" #
    # edge indices: "outlet first, inlet last" #
    ############################################

    ###################################
    # <Rule for concentration matrix> #
    # The order of variables is same  #
    # as the order of inlet number.   #
    ###################################

    #####################################################
    # <Rule for ini_list>                               #
    # The order of ini_list:                            #
    # "outlet first, inlet last."                       #
    # ["b or f", node, value, "i or o", m^3/s or Pa]    #
    #####################################################

    # %% Constants
    # Divide section to input constants
    w = 500 * 10 ** (-6)
    h = 500 * 10 ** (-6)
    eta = 1.002 * 10 ** (-3)
    n = 20001

    args = (eta, w, h, n)

    outlet_edge_idx = [0, 1, 2, 3, 4]  # outlet edge indices
    outlet_node_idx = [0, 1, 2, 3, 4]  # outlet node indices
    outlet_dict = {
        outlet_edge_idx[0]: outlet_node_idx[0],
        outlet_edge_idx[1]: outlet_node_idx[1],
        outlet_edge_idx[2]: outlet_node_idx[2],
        outlet_edge_idx[3]: outlet_node_idx[3],
        outlet_edge_idx[4]: outlet_node_idx[4],
    }  # outlet connection dictionary

    inlet_edge_idx = [15, 16]  # inlet edge indices
    inlet_node_idx = [14, 15]  # inlet node indices
    inlet_dict = {
        inlet_node_idx[0]: inlet_edge_idx[0],
        inlet_node_idx[1]: inlet_edge_idx[1],
    }  # inlet connection dictionary

    changing_edges = [i for i in range(17)]

    ini_list = [
        ["f", 0, np.nan, "o", "m^3/s"],
        ["f", 1, np.nan, "o", "m^3/s"],
        ["f", 2, np.nan, "o", "m^3/s"],
        ["f", 3, np.nan, "o", "m^3/s"],
        ["f", 4, np.nan, "o", "m^3/s"],
        ["f", 14, 10 * 10**-6 / 60, "i", "m^3/s"],
        ["f", 15, 10 * 10**-6 / 60, "i", "m^3/s"],
    ]

    plot_type = (
        "compare",
        "concentration",
        "difference",
    )  # str: choose plot type; 'compare', 'concentration', 'difference'

    nametag = "total"

    # %% Optimization of the popsize and ngen
    if "optimization" in whatToSolve:
        conc_wght = 0.5
        flow_wght = 1 - conc_wght
        optimization_functions(
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
        )

    # %% Changing lengths of the edges
    if "length" in whatToSolve:
        conc_wght = 0.5
        flow_wght = 1 - conc_wght
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
            popSize=600,
            nGen=600,
        )

        length_csv = "new_length_mat.csv"
        nametag = "revised"

    # %% Executing functions_all
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

    # %% Plotting graphs
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

    # %% Compare optimization
    # %% Compare optimization
    if "compare_optimization" in whatToSolve:
        whatToSolve = [
            "length",
            "source",
            "concentration",
            # "optimization",
            "compare_optimization",
        ]
        pareto_compare(
            whatToSolve,
            inc_csv,
            "new_lengths_mat.csv",
            conc_csv,
            args,
            ini_list,
            inlet_node_idx,
            inlet_edge_idx,
            outlet_node_idx,
            outlet_edge_idx,
            address,
        )


# %% Executing
if __name__ == "__main__":
    main()
