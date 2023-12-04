import numpy as np
import networkx as nx
from belief import Mode
from simulator import Simulator


def run_experiment(
        num_agents: int,
        num_sims: int,
        time_steps: int,
        mode: Mode,
        share_time_limit: float,
        delay: int,
        single_source: bool,
        same_partition: bool | None,
        graph,
        network_name: str,
        experiment: str,
        save: bool
):
    """
    Run simulation experiments...
    :param num_agents: number of agents in the network
    :param num_sims: number of simulations to run
    :param time_steps: number of time steps per simulation
    :param mode: set agent sharing mode
    :param share_time_limit: time an agent will share their newly attained beliefs; set np.inf for unlimited
    :param delay: time delay before retracted belief is added to model; set 0 for immediate addition
    :param single_source: retracted source same as false belief source (False only applied if delay > 0)
    :param same_partition: retracted source in same partition as fake (only random_partition_graph); set None for random
    :param graph: type of graph - distribution/density of network nodes
    :param network_name: name of graph network type
    :param experiment: name of output directory
    :param save: boolean; if True, saves plots to directory
    :return: None; saves output log file and visualizations if save is true
    """
    nx_params = {"n": num_agents, "k": 8, "p": 0.1}
    sub_experiment = f"{experiment}/Nodes={num_agents}_Steps={time_steps}_Simulations={num_sims}_Delay={delay}"

    # Run simulator
    sim = Simulator(
        num_agents=num_agents,
        num_sims=num_sims,
        time_steps=time_steps,
        graph=graph,
        nx_params=nx_params,
        sharing_mode=mode,
        share_time_limit=share_time_limit,
        delay=delay,
        single_source=single_source,
        same_partition=same_partition
    )
    avg_agent, sd_agent, frac_mean, frac_sd, belief_dist = sim.run_simulation(
        save=save,
        experiment=experiment,
        sub_experiment=sub_experiment,
        network_name=network_name,
        nx_params=nx_params
    )

    # Visualize output
    # Belief Over Time
    sim.vis_beliefs_over_time(
        data=(avg_agent, sd_agent, frac_mean, frac_sd),
        experiment=experiment,
        sub_experiment=sub_experiment,
        network_name=network_name,
        nx_params=nx_params,
        save=save,
        plot_sd=False
    )
    sim.vis_beliefs_over_time(
        data=(avg_agent, sd_agent, frac_mean, frac_sd),
        experiment=experiment,
        sub_experiment=sub_experiment,
        network_name=network_name,
        nx_params=nx_params,
        save=save,
        plot_sd=True
    )


if __name__ == '__main__':
    # Watts Strogatz Experiment - Small World
    run_experiment(
        num_agents=100,
        num_sims=1000,
        time_steps=1200,
        mode=Mode.Default,
        share_time_limit=np.inf,
        delay=0,
        single_source=False,
        same_partition=None,
        graph=nx.watts_strogatz_graph,
        network_name="SmallWorlds",
        experiment="Watts_Strogatz_Model",
        save=True,
    )

    # How to improve previous work ->
    # 1. Adjust idea transmission rate (100% to 94%) - double check %
    # 2. Try to make the graph grow over the time steps
    #       A. Maybe try iteratively getting subgraphs of subgraphs for each time point, then loop smallest to biggest to simulate growth
    # 3. Make it a directed graph where misinformation/retracted information can only travel 1 way
    #       A. Directed edges should only point 1 way, oldest nodes to newest nodes (direction information flows)

    # Previous ->

    # Graph & network structure
    # graph = nx.complete_graph
    # nx_params = {"n": N}
    # network_name = "complete"    # Set network name for output file

    # graph = nx.random_partition_graph
    # nx_params = {"sizes": [50, 50], "p_in": 0.4, "p_out": 0.2}  # Graph parameters
    # network_name = "RandomPartition"    # Set network name for output file

    # sim.visFinalBeliefDistributions(belief_dist=belief_dist,
    #                                 data=temporal_data,
    #                                 experiment=experiment,
    #                                 sub_experiment=sub_experiment,
    #                                 network_name=network_name,
    #                                 nx_params=nx_params,
    #                                 save=save)
