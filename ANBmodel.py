# %%
"""
Adaptive Networks of Beliefs Model
version 2026-06-01, Peter Steiglechner, steiglechner@csh.ac.at
"""

import networkx as nx
import numpy as np
import pandas as pd
import os
from itertools import combinations
from scipy.sparse import csr_matrix
import time
from joblib import Parallel, delayed
import multiprocessing
import igraph as ig

# FIXED PARAMETERS
M = 10
focal = 0  # dimension of the focal belief
ext_belief = focal
n_agents = 100
tau = 1
fixedBNat100 = False
two_external_events = False  # if True: sim runs to T=300 and there is a second external pressure event at t=200 
lam = 0.0
ext_time = list(np.arange(101, 151)) + (
    list(np.arange(201, 251)) if two_external_events else []
)
T = 200 if not two_external_events else 300
beforeRange = list(range(91, 101))
duringRange = list(range(141, 151))
afterRange = list(range(191, 201))
during2Range = list(range(241, 251))
after2Range = list(range(291, 301))

params_fixed = dict(
    M=M,
    focal=focal,
    n_agents=n_agents,
    tau=tau,
    ext_belief=ext_belief,
    ext_time=ext_time,
    T=T,
    beforeRange=[beforeRange[0], beforeRange[-1]],
    duringRange=[duringRange[0], duringRange[-1]],
    afterRange=[afterRange[0], afterRange[-1]],
    fixedBNat100=fixedBNat100,
    during2Range=(
        [during2Range[0], during2Range[-1]] if two_external_events else [None, None]
    ),
    after2Range=(
        [after2Range[0], after2Range[-1]] if two_external_events else [None, None]
    ),
    lam=lam,
)

belief_dimensions = list(range(0, M))
beliefupdate_order = belief_dimensions.copy()
belief_neighbours = {
    i: [nj for nj, j in enumerate(belief_dimensions) if j != i]
    for i in belief_dimensions
}
agentids = list(range(n_agents))
edge_list = list(combinations(belief_dimensions, 2))
edge_arr = np.array(edge_list)
edge_lookup = {c: nc for nc, c in enumerate(edge_list)}
adjacent_edge_ids = {
    dim: [
        edge_lookup[(dim, j)] if (dim, j) in edge_list else edge_lookup[(j, dim)]
        for j in belief_dimensions
        if j != dim
    ]
    for dim in belief_dimensions
}
belief_options = np.linspace(-1, 1, 21)
n_options = len(belief_options)


# PREPARE OUTPUT DATASET
meta_cols = ["t", "id"]
metric_cols = [
    "n_nbs",
    "Hpers",
    "Hpersfoc",
    "Hsoc",
    "Hext",
    "tb_tot",
    "tb_foc",
    "absOm_tot",
    "absOm_foc",
    "clust",
    "bc_foc",
    "expI",
    "x_focal",
    "extr_nonfoc",
]
response_cols = [
    "persistPos",
    "nonpersistPos",
    "compliant",
    "resilient",
    "resistant",
    "latecompliant",
]
columnIndex = len(meta_cols) + len(metric_cols) + len(response_cols)
beliefids = list(range(columnIndex, columnIndex + M))
edgeids = list(range(columnIndex + M, columnIndex + M + len(edge_list)))
delbeliefids = range(
    columnIndex + M + len(edge_list), columnIndex + M + len(edge_list) + tau * M
)
belief_cols = [str(b) for b in belief_dimensions]
edge_cols = [f"w{e[0]}{e[1]}" for e in edge_list]
columns = meta_cols + metric_cols + response_cols + belief_cols + edge_cols


def initialise_agents(init_w):
    edgeweights = [init_w for _ in edge_list]
    agent_list = []
    for id in range(n_agents):
        opinion_vector = np.random.choice(
            belief_options, size=M, p=np.ones(n_options) / n_options
        )
        agent_list.append(
            [0, id]
            + [np.nan] * (columnIndex - 2)
            + list(opinion_vector)
            + edgeweights
            + list(np.zeros(M)) * tau  # this will be used to store tau*M past belief changes
        )
    return np.array(agent_list)


def initialise_network(seed, link_prob):
    np.random.seed(seed)
    A = np.triu(
        (np.random.random((n_agents, n_agents)) <= link_prob).astype(bool),
        k=1,
    )
    A = A + A.T
    A_sparse = csr_matrix(A)
    nb_list = []
    for id in range(n_agents):
        nb_list.append(
            A_sparse[id].indices,
        )
    return nb_list


def glauber_fast(dim, agent, ext_strength, beta, rho, summed_social_beliefs):
    old_belief = agent[beliefids][dim]
    adjacent_beliefs = agent[beliefids][belief_neighbours[dim]]
    adjacent_edgeweights = agent[edgeids][adjacent_edge_ids[dim]]
    dH = -(belief_options - old_belief) * (
        np.sum(adjacent_edgeweights * adjacent_beliefs)
        + ext_strength * 1
        + rho * summed_social_beliefs
    )
    p = 1.0 / (1.0 + np.exp(beta * dH))
    return p / p.sum()


def update_edge_weights(agent, eps, mu, mean_social_edges):
    """Update all edge weights for an agent"""
    edgeweights = agent[edgeids]
    del_beliefs = agent[delbeliefids].reshape((tau, M)).mean(axis=0)
    lam = params_fixed["lam"]
    return (
        edgeweights
        + eps * del_beliefs[edge_arr[:, 0]] * del_beliefs[edge_arr[:, 1]]
        + mu * (mean_social_edges - edgeweights)
        - lam * edgeweights
    )


def update_belief(agent, ext_strength, beta, rho, summed_social_beliefs):
    """Update all belief values for an agent."""
    agent_prior_beliefs = np.copy(agent[beliefids])
    np.random.shuffle(beliefupdate_order)
    for dim in beliefupdate_order:
        probabilities = glauber_fast(
            dim,
            agent,
            0 if dim != focal else ext_strength,
            beta,
            rho,
            0 if dim != focal else summed_social_beliefs,
        )
        agent[beliefids[dim]] = np.random.choice(belief_options, p=probabilities)
    # shift del beliefs by one time step.
    agent[delbeliefids[M:]] = agent[delbeliefids[:-M]]
    # replace new del beliefs for last time step
    agent[delbeliefids[:M]] = agent[beliefids] - agent_prior_beliefs
    return agent


def get_energies(t, agents, nb_list, params):
    # Hpers = - 0.5 sum_i=1^M sum_j=1^M  w_ij * x_i x_j
    Hpers = [
        -0.5
        * np.sum(
            [
                agents[n][beliefids][dim]
                * np.sum(
                    agents[n][beliefids][belief_neighbours[dim]]
                    * agents[n][edgeids][adjacent_edge_ids[dim]]
                )  # this sums over belief_dims
                for dim in belief_dimensions  # second sum over belief_dims --> factor 0.5
            ]
        )
        for n in range(n_agents)
    ]
    Hpersfoc = [
        -agents[n][beliefids][focal]
        * np.sum(
            agents[n][beliefids][belief_neighbours[focal]]
            * agents[n][edgeids][adjacent_edge_ids[focal]]
        )
        for n in range(n_agents)
    ]
    Hext = (
        [0] * n_agents
        if t not in ext_time
        else [
            -params["ext_strength"] * 1 * agents[n][beliefids][focal]
            for n in range(n_agents)
        ]
    )
    Hsoc = [
        (
            -params["rho"]
            * agents[n][beliefids][focal]
            * np.nanmean(agents[nb_list[n]][:, beliefids[focal]])
            if len(nb_list[n]) > 0
            else 0
        )
        for n in range(n_agents)
    ]

    return Hpers, Hpersfoc, Hsoc, Hext


def get_metrics(agents):
    triangles = list(combinations(belief_dimensions, 3))
    tri_balance_tot = np.zeros(len(agents))
    tri_balance_foc = np.zeros(len(agents))
    for a, b, c in triangles:
        tri_balance = (
            agents[:, edgeids[edge_lookup[(a, b)]]]
            * agents[:, edgeids[edge_lookup[(a, c)]]]
            * agents[:, edgeids[edge_lookup[(b, c)]]]
        ) > 0
        tri_balance_tot += tri_balance
        if focal in [a, b, c]:
            tri_balance_foc += tri_balance

    bn_abs_meanedge_tot = np.abs(agents[:, edgeids]).mean(axis=1)
    bn_abs_meanedge_foc = np.abs(agents[:, edgeids][:, adjacent_edge_ids[focal]]).mean(
        axis=1
    )

    A = np.zeros((n_agents, M, M))
    for i, j in edge_list:
        A[:, i, j] = agents[:, edgeids][:, edge_lookup[(i, j)]]
        A[:, j, i] = A[:, i, j]
    absA = np.abs(A)
    w_max = absA.max(axis=(1,2), keepdims=True)  # (n_agents, 1, 1)
    absA_norm = absA /  w_max
    absA_norm_power = absA_norm ** (1/3)
    A_cubed_diag = np.einsum("aij,ajk,aki->ai", absA_norm_power, absA_norm_power, absA_norm_power) 
    degrees = (absA > 0).sum(axis=2)  # (n_agents, M)
    node_clustering = A_cubed_diag / (degrees * (degrees - 1))

    avg_weighted_clustering = np.nanmean(node_clustering, axis=1)   # (n_agents,)
    # focal_weighted_clustering = node_clustering[:, focal]            # (n_agents,)

    expected_influence = (A[:, focal, :] * agents[:, beliefids]).sum(axis=1)

    eps_val = 1e-6
    def bc_focal_for_agent(a):
        g = ig.Graph(n=M, edges=edge_list, directed=False)
        weights = [1.0 / (absA[a, i, j] + eps_val) if absA[a, i, j] > 0 else 1e9
                for i, j in edge_list]
        return g.betweenness(vertices=focal, weights=weights, directed=False)

    bc_foc = np.array([bc_focal_for_agent(a) for a in range(n_agents)])

    return (
        tri_balance_tot,
        tri_balance_foc,
        bn_abs_meanedge_tot,
        bn_abs_meanedge_foc,
        avg_weighted_clustering,
        bc_foc,
        expected_influence,
    )


def fill_metrics(t, agents, nb_list, params):
    agents[:, 2] = [len(nbs) for nbs in nb_list]
    Hpers, Hpersfoc, Hsoc, Hext = get_energies(t, agents, nb_list, params)
    tb_tot, tb_foc, absOm_tot, absOm_foc, clust, bc_foc, expI = get_metrics(agents)
    agents[:, 3] = Hpers
    agents[:, 4] = Hpersfoc
    agents[:, 5] = Hsoc
    agents[:, 6] = Hext
    agents[:, 7] = tb_tot
    agents[:, 8] = tb_foc
    agents[:, 9] = absOm_tot
    agents[:, 10] = absOm_foc
    agents[:, 11] = clust
    agents[:, 12] = bc_foc
    agents[:, 13] = expI
    agents[:, 14] = agents[:, beliefids[focal]]
    agents[:, 15] = np.mean(
        np.abs(agents[:, [k for k in beliefids if not k == beliefids[focal]]]), axis=1
    )
    return agents


def get_output(snapshots):
    beliefs = pd.DataFrame(
        snapshots[:, [0, 1, beliefids[focal]]], columns=["t", "id", "x_foc"]
    )
    pivot = beliefs.pivot_table(index="t", columns="id", values="x_foc")
    before = pivot.loc[pivot.index.isin(beforeRange)].mean(axis=0)
    during = pivot.loc[pivot.index.isin(duringRange)].mean(axis=0)
    after = pivot.loc[pivot.index.isin(afterRange)].mean(axis=0)
    out = snapshots[:, : (columnIndex + M + len(edge_list))]
    dfFull = pd.DataFrame(out, columns=columns)
    if detail:
        dfFull.loc[dfFull.index[-n_agents:], "persistPos"] = (
            ((before >= 0) * (during >= 0) * (after >= 0)).astype(int).values
        )  # persistent positive
        dfFull.loc[dfFull.index[-n_agents:], "nonpersistPos"] = (
            ((before >= 0) * (after < 0)).astype(int).values
        )  # nonpersistent positive
        dfFull.loc[dfFull.index[-n_agents:], "compliant"] = (
            ((before < 0) * (during >= 0) * (after >= 0)).astype(int).values
        )  # compliant
        dfFull.loc[dfFull.index[-n_agents:], "resilient"] = (
            ((before < 0) * (during >= 0) * (after < 0)).astype(int).values
        )  # resilient
        dfFull.loc[dfFull.index[-n_agents:], "resistant"] = (
            ((before < 0) * (during < 0) * (after < 0)).astype(int).values
        )  # resistant
        dfFull.loc[dfFull.index[-n_agents:], "latecompliant"] = (
            ((before < 0) * (during < 0) * (after > 0)).astype(int).values
        )  # late-compliant
    else:
        snapzerodf = dfFull.loc[dfFull.t == 0][
            meta_cols + metric_cols + belief_cols + edge_cols
        ]
        snapbeforedf = dfFull.loc[dfFull.t == 100][
            meta_cols + metric_cols + belief_cols + edge_cols
        ]
        snapfinaldf = dfFull.loc[dfFull.t == 200][
            meta_cols + metric_cols + belief_cols + edge_cols
        ]
        beforedf = (
            dfFull.loc[dfFull.t.isin(beforeRange)]
            .groupby("id")[metric_cols + belief_cols + edge_cols]
            .mean()
            .reset_index()
        )
        duringdf = (
            dfFull.loc[dfFull.t.isin(duringRange)]
            .groupby("id")[metric_cols + belief_cols + edge_cols]
            .mean()
            .reset_index()
        )
        afterdf = (
            dfFull.loc[dfFull.t.isin(afterRange)]
            .groupby("id")[metric_cols + belief_cols + edge_cols]
            .mean()
            .reset_index()
        )
        beforedf["t"] = np.mean(beforeRange)
        duringdf["t"] = np.mean(duringRange)
        afterdf["t"] = np.mean(afterRange)
        if two_external_events:
            during2df = (
                dfFull.loc[dfFull.t.isin(during2Range)]
                .groupby("id")[metric_cols + belief_cols + edge_cols]
                .mean()
                .reset_index()
            )
            after2df = (
                dfFull.loc[dfFull.t.isin(after2Range)]
                .groupby("id")[metric_cols + belief_cols + edge_cols]
                .mean()
                .reset_index()
            )
            snapfinal2df = dfFull.loc[dfFull.t == 300][
                meta_cols + metric_cols + belief_cols + edge_cols
            ]
            during2df["t"] = np.mean(during2Range)
            after2df["t"] = np.mean(after2Range)

        # save response dummies of agents only at one time step.
        for dff in [beforedf]:
            dff["persistPos"] = (
                (before >= 0) * (during >= 0) * (after >= 0)
            )  # persistent positive
            dff["nonpersistPos"] = (before >= 0) * (after < 0)  # nonpersistent positive
            dff["compliant"] = (before < 0) * (during >= 0) * (after >= 0)  # compliant
            dff["resilient"] = (before < 0) * (during >= 0) * (after < 0)  # resilient
            dff["resistant"] = (before < 0) * (during < 0) * (after < 0)  # resistant
            dff["latecompliant"] = (
                (before < 0) * (during < 0) * (after > 0)
            )  # late-compliant
        df = pd.concat(
            [snapzerodf, beforedf, snapbeforedf, duringdf, afterdf, snapfinaldf]
            + ([during2df, after2df, snapfinal2df] if two_external_events else [])
        )
    return dfFull if detail else df


def run_simulation(params):
    np.random.seed(params["seed"])
    eps, mu, beta, rho, ext_strength = (
        params["eps"],
        params["mu"],
        params["beta"],
        params["rho"],
        params["ext_strength"],
    )
    agents = initialise_agents(params["init_w"])
    nb_list = initialise_network(params["seed"], params["link_prob"])

    # get metrics
    agents = fill_metrics(0, agents, nb_list, params)
    snapshots = np.copy(agents)  # initialise snapshots with the agent list at t=0

    # Main simulation loop
    time_steps = np.arange(0, T + 1, 1)
    for t in time_steps[1:]:
        curr_ext_strength = ext_strength if t in ext_time else 0
        np.random.shuffle(agentids)
        for n in agentids:
            social_beliefs = agents[nb_list[n]][:, beliefids[focal]]
            summed_social_beliefs = (
                0 if len(social_beliefs) == 0 else np.sum(social_beliefs)
            )
            agents[n] = update_belief(
                agents[n], curr_ext_strength, beta, rho, summed_social_beliefs
            )
            if fixedBNat100 and (t >= ext_time[0]):
                pass
            else:
                mu_c = 0 if len(nb_list[n]) == 0 else mu
                mu_c = (
                    mu_c if (not params["fixedBNat100"] or not t > ext_time[0]) else 0
                )
                eps_c = (
                    eps if (not params["fixedBNat100"] or not t > ext_time[0]) else 0
                )
                mean_social_edges = (
                    0
                    if len(nb_list[n]) == 0
                    else agents[nb_list[n]][:, edgeids].mean(axis=0)
                )
                agents[n, edgeids] = update_edge_weights(
                    agents[n], eps_c, mu_c, mean_social_edges
                )
        agents[:, 0] = t
        if t in track_times or t == time_steps[-1]:
            agents = fill_metrics(t, agents, nb_list, params)
            snapshots = np.concatenate([snapshots, agents])

    output_df = get_output(snapshots)
    return output_df


def run_one(seed, link_prob, init_w, beta, rho, eps, mu, fixedBNat100, ext_strength):

    # results_folder = (
    #     f"{time.gmtime().tm_year}-{time.gmtime().tm_mon:02d}-{time.gmtime().tm_mday:02d}"
    #     + "_simOut/"
    # )
    results_folder = "simOut/"
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)
    if detail and not os.path.isdir(results_folder + "detailed/"):
        os.mkdir(results_folder + "detailed/")
    params = {
        "link_prob": link_prob,
        "init_w": init_w,
        "beta": beta,
        "rho": rho,
        "eps": eps,
        "mu": mu,
        "fixedBNat100": fixedBNat100,
        "ext_strength": ext_strength,
        "seed": seed,
    }

    fname = (
        results_folder
        + ("detailed/" if detail else "")
        + "sim_"
        + "_".join(
            [
                (
                    f"{k}{v}"  # integer params
                    if k in ["seed", "ext_strength"]
                    else (
                        f"{k}{v:.3f}"
                        if k == "mu"
                        else (
                            (k if v else "") if k == "fixedBNat100" else f"{k}{v:.2f}"
                        )
                    )
                )
                for k, v in params.items()
            ]
        )
    )
    fname = fname.replace("__", "_")
    fname += "" if not two_external_events else "_2events"
    fname += f"_lambda{lam:.4f}" if lam > 0 else ""
    if (seed % 25) == 0:
        print(fname + "...")

    if True:  # not os.path.isfile(fname+".csv"):
        out = run_simulation(params)

        for k, v in params.items():
            out[k] = v

        out.to_csv(fname + ("_detailed" if detail else "") + ".csv")
    return fname


# %%
# Main execution
if __name__ == "__main__":
    link_prob = 10 / n_agents
    # init_w = 0.2
    beta = 3.0
    rho = 1.0 / 3.0
    # eps = 1.0
    # mu = 0.0
    # ext_strength = 4

    param_combis = [
        [link_prob, init_w, beta, rho, eps, mu, fixedBNat100]
        for init_w, eps, mu, fixedBNat100 in [
            (0.2, 0.0, 0.0, False),    # static, low edge weights
            (0.8, 0.0, 0.0, False),    # static, high edge weights
            (0.2, 1.0, 0.0, True),    # adaptive until external pressure
            (0.2, 1.0, 0.0, False),    # fully adaptive, strong edge weights
        ]
    ]
    
    # eps = 1.0
    # init_w=0.2
    # fixedBNat100 = False
    # param_combis = [
    #     [link_prob, init_w, beta, rho, eps, mu, fixedBNat100]
    #     for mu in [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5, 1.0
    #     ]
    # ]

    pressures =  [0, 1, 2, 4, 8, 16]

    param_combis = [
        p + [ext_strength] for p in param_combis for ext_strength in pressures
    ]

    seeds = list(range(0, 10))  ## TODO increase

    detail = False
    track_times = (
        np.arange(T + 1)
        if detail
        else [0, 1]
        + beforeRange
        + [100]
        + duringRange
        + afterRange
        + [200]
        + ((during2Range + after2Range + [300]) if two_external_events else [])
    )

    if detail and len(seeds) > 10:
        print("...this will take very long. Reduce the number of seeds or set detail to False. Aborting")
        quit()
    else:
        param_combis_withSeed = [
            param_combi + [seed] for param_combi in param_combis for seed in seeds
        ]
        print("running on jobs: ", multiprocessing.cpu_count() - 1)

        Parallel(n_jobs=max(1, multiprocessing.cpu_count() - 1))(
            delayed(run_one)(
                seed, link_prob, init_w, beta, rho, eps, mu, fixedBNat100, ext_strength
            )
            for link_prob, init_w, beta, rho, eps, mu, fixedBNat100, ext_strength, seed in param_combis_withSeed
        )


# %%
