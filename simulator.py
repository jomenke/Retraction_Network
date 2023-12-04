import numpy as np
import pandas as pd
from belief import Mode
from model import KnowledgeModel
import matplotlib.pyplot as plt
import os
from networkx import Graph


class Simulator:
    """A simulator to run multiple knowledge models."""
    def __init__(
            self,
            num_agents: int,
            num_sims: int,
            time_steps: int,
            graph,
            nx_params: dict,
            sharing_mode: Mode,
            share_time_limit: float,
            delay: int,
            single_source: bool,
            same_partition: bool | None
    ):
        """
        Initialize a Simulator class.
        :param num_agents: number of agents in the network
        :param num_sims: number of simulations to run
        :param time_steps: number of time steps per simulation
        :param graph: a type of Networkx graph - distribution/density of network nodes
        :param nx_params: dictionary of keyword arguments to create a Networkx graph
        :param sharing_mode: predefined in Mode class; integer mapped to a specific mode (e.g., default = 0)
        :param share_time_limit: a float indicating the time limit within which new beliefs are shared; np.inf if endless
        :param delay: Time delay before retracted belief is added to model; set 0 for immediate addition
        :param single_source: boolean; retracted source same as false belief source (False only applied if delay > 0)
        :param same_partition: retracted and fake source in same partition if random_partition_graph; when None, random
        """
        self.num_agents = num_agents
        self.num_sims = num_sims
        self.time_steps = time_steps
        self.graph = graph
        self.nx_params = nx_params
        self.mode = sharing_mode
        self.delay = delay
        self.single_source = single_source
        self.same_partition = same_partition
        self.share_time_limit = share_time_limit

    def run_model(self, network: Graph) -> tuple[dict, dict, list[set]]:
        """
        Run network model for number of time-steps and return logs.
        :param network: a Networkx graph
        :return: a tuple containing 3 elements: belief history, interaction history, and pair history
        """

        # create model
        model = KnowledgeModel(network=network,
                               sharing_mode=self.mode,
                               share_time=self.share_time_limit,
                               delay=self.delay,
                               single_source=self.single_source,
                               same_partition=self.same_partition)

        # run model
        for t in range(self.time_steps - 1):
            model.step()

        return model.logs()

    def run_simulation(
            self,
            save: bool,
            experiment: str,
            sub_experiment: str,
            network_name: str,
            nx_params: dict
    ) -> tuple:
        """
        Run model for number of simulations and return aggregate logs.
        :param save: boolean; if True, saves plots to directory
        :param experiment: name of output directory
        :param sub_experiment: detailed name of folder in output directory
        :param network_name: name of graph network type
        :param nx_params: dictionary of keyword arguments to create a Networkx graph
        :return: a tuple containing avg and std per agent, avg and std fraction belief, and the belief distribution
        """

        num_neutral_per_agent = np.empty(shape=self.num_sims)
        num_fake_per_agent = np.empty(shape=self.num_sims)
        num_retracted_per_agent = np.empty(shape=self.num_sims)
        neutral_per_timestep = np.empty(shape=(self.num_sims, self.time_steps))
        fake_per_timestep = np.empty(shape=(self.num_sims, self.time_steps))
        retracted_per_timestep = np.empty(shape=(self.num_sims, self.time_steps))

        directory = "./output/" + experiment + "/" + sub_experiment + "/data/"
        name = network_name + '_' + '_'.join(['{}={}'.format(k, v) for k, v in nx_params.items()])
        path = directory + "N{N}-T{T}-S{S}-{shr}-{dly}-{name}-data.csv".format(
            N=self.num_agents, T=self.time_steps, S=self.num_sims, shr=self.share_time_limit, dly=self.delay, name=name
        )

        for s in range(self.num_sims):
            # run model
            network = self.graph(**self.nx_params)  # generate network from graph and params
            logs = self.run_model(network=network)
            df_belief = pd.DataFrame.from_dict(logs[0])

            if save:  # write raw data to output directory
                if not os.path.exists(directory):
                    os.makedirs(directory)

                out = pd.DataFrame(
                    index=[x for x in range(self.time_steps)], columns=['s', 't'] + [x for x in range(self.num_agents)]
                )
                # vector of time-steps
                out.iloc[:, 0] = np.repeat(s + 1, repeats=self.time_steps)
                # time-step number
                out.iloc[:, 1] = np.linspace(start=1, stop=self.time_steps, num=self.time_steps, dtype=int)
                out.iloc[:, 2:self.num_agents + 2] = df_belief.values
                out.to_csv(path,
                           index=False,
                           header=True if s == 0 else False,
                           mode='a',  # append df to csv
                           encoding='utf-8')

            # eval output
            num_neutral_per_agent[s] = np.mean(np.sum(df_belief.values == 0, axis=0))
            num_fake_per_agent[s] = np.mean(np.sum(df_belief.values == 1, axis=0))
            num_retracted_per_agent[s] = np.mean(np.sum(df_belief.values == 2, axis=0))
            neutral_per_timestep[s, :] = np.mean(df_belief.values == 0, axis=1)
            fake_per_timestep[s, :] = np.mean(df_belief.values == 1, axis=1)
            retracted_per_timestep[s, :] = np.mean(df_belief.values == 2, axis=1)

        # aggregate beliefs over time
        neutral_per_agent_avg = np.mean(num_neutral_per_agent)
        neutral_per_agent_sd = np.std(num_neutral_per_agent)
        fake_per_agent_avg = np.mean(num_fake_per_agent)
        fake_per_agent_sd = np.std(num_fake_per_agent)
        retracted_per_agent_avg = np.mean(num_retracted_per_agent)
        retracted_per_agent_sd = np.std(num_retracted_per_agent)
        frac_neutral_per_timestep = np.mean(neutral_per_timestep, axis=0)
        frac_neutral_per_timestep_sd = np.std(neutral_per_timestep, axis=0)
        frac_fake_per_timestep = np.mean(fake_per_timestep, axis=0)
        frac_fake_per_timestep_sd = np.std(fake_per_timestep, axis=0)
        frac_retracted_per_timestep = np.mean(retracted_per_timestep, axis=0)
        frac_retracted_per_timestep_sd = np.std(retracted_per_timestep, axis=0)

        # aggregate final belief distributions
        neutral_dist = neutral_per_timestep[:, self.time_steps - 1]
        fake_dist = fake_per_timestep[:, self.time_steps - 1]
        retracted_dist = retracted_per_timestep[:, self.time_steps - 1]

        # bundle aggregated output
        avg_per_agent = (neutral_per_agent_avg, fake_per_agent_avg, retracted_per_agent_avg)
        sd_per_agent = (neutral_per_agent_sd, fake_per_agent_sd, retracted_per_agent_sd)
        frac_belief_mean = (frac_neutral_per_timestep, frac_fake_per_timestep, frac_retracted_per_timestep)
        frac_belief_sd = (frac_neutral_per_timestep_sd, frac_fake_per_timestep_sd, frac_retracted_per_timestep_sd)
        belief_dist = (neutral_dist, fake_dist, retracted_dist)

        return avg_per_agent, sd_per_agent, frac_belief_mean, frac_belief_sd, belief_dist

    def vis_final_belief_distributions(
            self,
            belief_dist: tuple,
            data: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
            experiment: str,
            sub_experiment: str,
            network_name: str,
            nx_params: dict,
            save: bool
    ) -> None:
        """
        Plot belief distributions on final time step over all simulations.
        :param belief_dist: a tuple containing the distribution of the different beliefs
        :param data: a tuple containing 4 tuples with 3 arrays (one for each belief) - [mean, std, frac_mean, frac_std]
        :param experiment: name of output directory
        :param sub_experiment: detailed name of folder in output directory
        :param network_name: name of graph network type
        :param nx_params: dictionary of keyword arguments to create a Networkx graph
        :param save: boolean; if True, saves plots to directory
        :return: None
        """

        avg_agent_beliefs, sd_agent_beliefs, _, _ = data
        avg_neutral, avg_fake, avg_retracted = avg_agent_beliefs
        sd_neutral, sd_fake, sd_retracted = sd_agent_beliefs
        neutral_dist, fake_dist, retracted_dist = belief_dist
        ranges = np.linspace(start=0, stop=1, num=100)

        plt.subplot(3, 1, 1)
        plt.hist(neutral_dist, bins=ranges)
        plt.ylim(ymin=0, ymax=self.num_sims)
        plt.ylabel("Neutral")

        plt.subplot(3, 1, 2)
        plt.hist(fake_dist, bins=ranges)
        plt.ylim(ymin=0, ymax=self.num_sims)
        plt.ylabel("Fake")

        plt.subplot(3, 1, 3)
        plt.hist(retracted_dist, bins=ranges)
        plt.ylim(ymin=0, ymax=self.num_sims)
        plt.xlabel("Fraction of population holding belief at time T")
        plt.ylabel("Retracted")

        if save:  # write plot to output directory
            directory = "./output/" + experiment + "/" + sub_experiment + "/hist/"
            if not os.path.exists(directory):
                os.makedirs(directory)
            network_name = network_name + '_' + '_'.join(['{}={}'.format(k, v) for k, v in nx_params.items()])
            filename = ("/N={N}-T={T}-S={S}-shr={shr}-dly={dly}-avg_n={avg_n}(sd={sd_n})-"
                        "avg_f={avg_f}(sd={sd_f})-avg_r={avg_r}(sd={sd_r})-{name}.png").format(
                N=self.num_agents, T=self.time_steps, S=self.num_sims, shr=self.share_time_limit, dly=self.delay, name=network_name,
                avg_n=round(avg_neutral, 1), avg_f=round(avg_fake, 1), avg_r=round(avg_retracted, 1),
                sd_n=round(sd_neutral, 1), sd_f=round(sd_fake, 1), sd_r=round(sd_retracted, 1))
            plt.savefig(directory + filename, bbox_inches="tight")

        plt.show()


    def vis_beliefs_over_time(
            self,
            data: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
            experiment: str,
            sub_experiment: str,
            network_name: str,
            nx_params: dict,
            save: bool,
            plot_sd: bool = False
    ) -> None:
        """
        Plot data and output to file.
        :param data: a tuple containing 4 tuples with 3 arrays (one for each belief) - [mean, std, frac_mean, frac_std]
        :param experiment: name of output directory
        :param sub_experiment: detailed name of folder in output directory
        :param network_name: name of graph network type
        :param nx_params: dictionary of keyword arguments to create a Networkx graph
        :param save: boolean; if True, saves plots to directory
        :param plot_sd: boolean; if True, add standard deviation to the plot
        :return: None
        """

        # unpack aggregate data
        avg_num, sd_num, frac_belief_mean, frac_belief_sd = data
        avg_neutral, avg_fake, avg_retracted = avg_num
        sd_neutral, sd_fake, sd_retracted = sd_num
        neutral_mean, fake_mean, retracted_mean = frac_belief_mean
        neutral_sd, fake_sd, retracted_sd = frac_belief_sd

        alpha = 0.5
        plt.plot(range(self.time_steps), fake_mean, label="False", color="tab:red", ls="-")
        plt.plot(range(self.time_steps), neutral_mean, label="Neutral", color="tab:orange", ls="-")
        plt.plot(range(self.time_steps), retracted_mean, label="Retracted", color="tab:green", ls="-")
        if plot_sd:
            plt.plot(range(self.time_steps), fake_mean + fake_sd, color="tab:red", ls="--", alpha=alpha)
            plt.plot(range(self.time_steps), fake_mean - fake_sd, color="tab:red", ls="--", alpha=alpha)
            plt.plot(range(self.time_steps), neutral_mean + neutral_sd, color="tab:orange", ls="--", alpha=alpha)
            plt.plot(range(self.time_steps), neutral_mean - neutral_sd, color="tab:orange", ls="--", alpha=alpha)
            plt.plot(range(self.time_steps), retracted_mean + retracted_sd, color="tab:green", ls="--", alpha=alpha)
            plt.plot(range(self.time_steps), retracted_mean - retracted_sd, color="tab:green", ls="--", alpha=alpha)
        plt.xlim(0, self.time_steps)
        plt.ylim(0, 1.11)
        plt.xlabel("Time")
        plt.ylabel("Proportion of population holding belief")
        plt.legend(loc="lower center", ncol=3, fancybox=True, bbox_to_anchor=(0.5, 0.9))

        if save:  # write plot to output directory
            directory = "./output/" + experiment + "/" + sub_experiment
            directory += "/sd" if plot_sd else ""  # create sub_folder for sd plots
            if not os.path.exists(directory):
                os.makedirs(directory)
            network_name = network_name + '_' + '_'.join(['{}={}'.format(k, v) for k, v in nx_params.items()])
            filename = ("/N={N}-T={T}-S={S}-shr={shr}-dly={dly}-avg_n={avg_n}(sd={sd_n})-"
                        "avg_f={avg_f}(sd={sd_f})-avg_r={avg_r}(sd={sd_r})-{name}{sd}.png").format(
                N=self.num_agents, T=self.time_steps, S=self.num_sims, shr=self.share_time_limit, dly=self.delay, name=network_name,
                avg_n=round(avg_neutral, 1), avg_f=round(avg_fake, 1), avg_r=round(avg_retracted, 1),
                sd_n=round(sd_neutral, 1), sd_f=round(sd_fake, 1), sd_r=round(sd_retracted, 1),
                sd=("-sd" if plot_sd else ""))
            plt.savefig(directory + filename, bbox_inches="tight")

        plt.show()
