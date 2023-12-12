from __future__ import annotations
from typing import TYPE_CHECKING
from mesa.time import BaseScheduler
from simpleLogger import SimpleLogger
import random
from barabasi_albert_graph_modified import _random_subset


if TYPE_CHECKING:
    # To prevent cyclic dependency...
    from agent import Article
    from model import AcademicLiterature


class SimpleActivation(BaseScheduler):
    """
    A simple scheduler which randomly picks only one pair of agents,
    activating their step function.
    """
    def __init__(self, model: AcademicLiterature):
        """
        Initialize a simple scheduler.
        :param model: a AcademicLiterature defining the simulation entities
        """
        super().__init__(model)

        # create logger
        self.logger = SimpleLogger(model)

    def add(self, agent: Article) -> None:
        """
        Add an Agent object to the schedule and logger.
        :return: None
        """
        self._agents[agent.unique_id] = agent
        self.logger.add(agent)

    def logs(self) -> tuple[dict, dict, list[set]]:
        """
        Wrapper for logs function in SimpleLogger class.
        Get agents' belief and interaction history, respectively.
        :return: a tuple containing 3 elements: belief history, interaction history, and pair history
        """
        return self.logger.logs()

    def choose(self) -> tuple[Article, Article]:
        """
        Randomly chooses a single pair of neighboring agents for interaction.
        :return: a tuple containing 2 Article instances
        """
        keys = list(self._agents.keys())
        # key_a = random.choice(keys) # TODO: change to more efficient random
        agent_a, agent_b = None, None
        try:
            # pick agent A
            key_a = list(_random_subset(keys, 1))
            agent_a = self.model.schedule.agents[key_a[0]]
            # pick agent B
            key_b = random.choice(agent_a.cited_by)  # TODO: change to cited_by
            agent_b = self.model.schedule.agents[key_b]
        except (KeyError, IndexError) as e:
            pass

        return agent_a, agent_b

    def step(self) -> None:
        """
        Increments the timer for all agents, then lets one pair of agents interact.
        :return: None
        """
        # increment timers for agents in AcademicLiterature
        for agent in self.agents:
            agent.tick()

        # steps --
        # check if false is there
        # loop through agents until we hit a false article
        # get articles to transmit based on citation network
        cited_by = list(self._agents.keys())


        # old ---
        agent_a, agent_b = self.choose()

        # interact
        agent_a.step(agent_b)
        agent_b.step(agent_a)

        # log results
        self.logger.log(agent_a, agent_b)

        # increment counters
        self.steps += 1
        self.time += 1
