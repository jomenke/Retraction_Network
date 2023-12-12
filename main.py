import numpy as np
import networkx as nx
from belief import Mode
from simulator import Simulator
from barabasi_albert_graph_modified import barabasi_albert_graph_modified


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
        nx_params: dict,
        custom_network: bool,
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
    :param nx_params: dict - parameters for networkx graph
    :param custom_network: boolean - if True, utilize custom network
    :param network_name: str - name of graph network type
    :param experiment: str - name of output directory
    :param save: boolean - if True, saves plots to directory
    :return: None - saves output log file and visualizations if save is true
    """

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
        same_partition=same_partition,
        custom_network=custom_network,
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
    # Final Belief Distribution
    sim.vis_final_belief_distributions(
        belief_dist=belief_dist,
        data=(avg_agent, sd_agent, frac_mean, frac_sd),
        experiment=experiment,
        sub_experiment=sub_experiment,
        network_name=network_name,
        nx_params=nx_params,
        save=save
    )


if __name__ == '__main__':
    # Watts Strogatz Experiment - Small World
    # run_experiment(
    #     num_agents=100,
    #     num_sims=1000,
    #     time_steps=1200,
    #     mode=Mode.Default,
    #     share_time_limit=np.inf,
    #     delay=0,
    #     single_source=False,
    #     same_partition=None,
    #     graph=nx.watts_strogatz_graph,
    #     nx_params = {"n": num_agents, "k": 26, "p": 0.1},
    #     custom_network=False,
    #     network_name="SmallWorlds",
    #     experiment="Watts_Strogatz_Model",
    #     save=True,
    # )

    # Custom Network - SmallWorld + Preferential Attachment
    run_experiment(
        num_agents=4000,
        num_sims=1,
        time_steps=5000,
        mode=Mode.Default,
        share_time_limit=np.inf,
        delay=0,
        single_source=False,
        same_partition=None,
        graph=barabasi_albert_graph_modified,
        nx_params={"n": 4000},
        custom_network=True,
        network_name="CitationNetwork",
        experiment="Watts_Strogatz_Preferential_Attachment_Model",
        save=True,
    )
