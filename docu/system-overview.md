# System Overview

How the pieces of the repository fit together: a parameter sweep in `ANBmodel.py` writes one CSV per
simulation, and five independent analysis scripts read those CSVs and produce the paper's figures and
tables. For details, see [running-simulations.md](running-simulations.md), [model.md](model.md) and
[analysis.md](analysis.md).

## Pipeline

```mermaid
flowchart LR
    subgraph CFG["Configuration (edit ANBmodel.py)"]
        direction TB
        C1["Module constants<br/>M, n_agents, tau, lam,<br/>two_external_events"]
        C2["Main block<br/>param_combis × pressures × seeds<br/>detail = True / False"]
    end

    subgraph SIM["ANBmodel.py"]
        direction TB
        P["joblib.Parallel<br/>cpu_count() − 1 workers"]
        R["run_one(...)<br/>builds filename from params"]
        S["run_simulation(params)<br/>one seed, one config"]
        P -->|"one job per<br/>combination"| R --> S
    end

    subgraph OUT["simOut/ (git-ignored)"]
        direction TB
        E[("Ensemble CSVs<br/>sim_..._seed{n}.csv<br/>34 tracked time steps")]
        D[("Detailed CSVs<br/>detailed/sim_..._detailed.csv<br/>all 201 time steps")]
    end

    subgraph ANA["Analysis scripts (independent, # %% cells)"]
        direction TB
        A1["analyse_timeseries.py"]
        A2["analyse_responseFrequencies.py"]
        A3["analyse_BNmetrics.py"]
        A4["analyse_socialAdaptation.py"]
        A5["analyse_sensitivity.py"]
    end

    subgraph RES["Results"]
        direction TB
        F[("figs/*.png, *.pdf<br/>Fig. 2–6, A1–A8")]
        T["LaTeX tables on stdout<br/>Table 3, A1–A7"]
    end

    C1 --> SIM
    C2 --> P
    S -->|"detail = False"| E
    S -->|"detail = True"| D

    D --> A1
    D --> A2
    E --> A2
    E --> A3
    E --> A4
    E --> A5

    A1 --> F
    A2 --> F
    A3 --> F
    A3 --> T
    A4 --> F
    A5 --> F
```

The analysis scripts don't import anything from `ANBmodel.py`. They find their input files by exact
filename, so the filename that `run_one` builds is the only link between the two halves. If a parameter
appears in the filename with a different format, or a run is missing, the analysis script fails when it
calls `pd.read_csv`. For the runs each script expects, see
[running-simulations.md](running-simulations.md#runs-needed-for-each-analysis-script).

## Inside one simulation

```mermaid
flowchart TD
    I["initialise_agents(init_w)<br/>M = 10 beliefs + 45 edge weights per agent"]
    N["initialise_network(seed, link_prob)<br/>random social network → nb_list"]
    M0["fill_metrics(t = 0)"]
    I --> N --> M0 --> LOOP

    subgraph LOOP["for t = 1 … T (200)"]
        direction TB
        X{"t in ext_time<br/>(101–150)?"}
        X -->|yes| XS["external pressure = s"]
        X -->|no| X0["external pressure = 0"]
        XS --> SH
        X0 --> SH
        SH["shuffle agent order"]

        subgraph AG["for each agent n"]
            direction TB
            B["update_belief<br/>Glauber step (β) on personal +<br/>social + external dissonance"]
            W{"fixedBNat100 and<br/>t ≥ 101?"}
            EW["update_edge_weights<br/>internal (ε, Hebbian) +<br/>social (μ, neighbour mean BN)"]
            B --> W
            NA["next agent"]
            W -->|no| EW --> NA
            W -->|"yes: BN frozen"| NA
        end

        SH --> AG
        AG --> TR{"t in track_times?"}
        TR -->|yes| FM["fill_metrics(t)<br/>append snapshot"]
        TR -->|no| NT["next t"]
        FM --> NT
    end

    LOOP --> GO["get_output(snapshots)<br/>metrics, before/during/after averages,<br/>response classification"]
    GO --> CSV[("CSV in simOut/")]
```

In ensemble mode, `track_times` holds only t = 0, 1, 91–100, 141–150, 191–200 and 200. In detailed mode it
holds every step. `get_output` adds the averaged rows `t = 95.5` (before), `145.5` (during) and `195.5`
(after). Only the `95.5` row carries the response dummies (compliant, resilient, resistant, …) that the
analysis scripts count. See [model.md](model.md) for the update equations and the response definitions.
