from mesa import Model
from agent import Article
from simpleScheduler import SimpleActivation
import random
from belief import Belief, Mode
import networkx as nx
import numpy as np


class AcademicLiterature(Model):
    """
    A model with some number of agents.
    >>> model = AcademicLiterature(nx.watts_strogatz_graph(n=100, k=8, p=0.1), Mode.Default, 1.0, 0)
    >>> model.step()
    >>> logs_ = model.logs()
    >>> len(logs_)  # len(logs_) = number of classes (neutral, false, retracted)
    3
    >>> isinstance(logs_[0], dict)
    True
    >>> isinstance(logs_[2], list)
    True
    >>> article = Article(1, model, [], 1.0)
    >>> article.is_sharing()
    True
    >>> article.tick()
    >>> article.clock
    1
    >>> article.belief_time
    1
    >>> article.tick()
    >>> article.is_sharing()
    False
    >>> article_1 = Article(2, model, [], np.inf)
    >>> article_1.set_belief(Belief.Fake)
    >>> article_2 = Article(2, model, [article_1], np.inf)
    >>> article_2.set_belief(Belief.Retracted)
    >>> article_1.update(article_2)
    >>> article_1.belief
    <Belief.Retracted: 2>
    """
    def __init__(
            self,
            network,
            sharing_mode: Mode,
            share_time: float,
            delay: int,
            single_source: bool = False,
            same_partition: bool | None = True
    ):
        """
        Initialize a AcademicLiterature.
        :param network: a NetworkX graph instance
        :param sharing_mode: predefined in Mode class; integer mapped to a specific mode (e.g., default = 0)
        :param share_time: a float indicating the time limit within which new beliefs are shared; np.inf if endless
        :param delay: time delay before retracted belief is added to model; set 0 for immediate addition
        :param single_source: boolean; retracted source same as false belief source (False only applied if delay > 0)
        :param same_partition: retracted and fake source in same partition if random_partition_graph; when None, random
        """
        super().__init__()  # new - remove if everything still works as expected

        self.mode = sharing_mode
        self.G = network
        self.num_agents = self.G.number_of_nodes()
        self.schedule = SimpleActivation(self)
        self.delay = delay
        self.single_source = single_source
        self.same_partition = same_partition
        self.misinformation = False

        # Create agents
        for i in range(self.num_agents):
            cited_by = list(self.G.cited_by(i))
            a = Article(unique_id=i, model=self, cited_by=cited_by, share_time=share_time)

            # TODO: change this to randomly add it to well connected vs. less connected nodes
            if i == 0:
                # Give Agent 0 false information
                a.belief = Belief.Fake
                self.agentZero = a
            if (i == 1) and (self.delay == 0) and self.same_partition is None and self.single_source:
                # Give Agent 1 true information
                a.belief = Belief.Retracted
                self.misinformation = True

            self.schedule.add(a)

        if (self.delay == 0) and (self.same_partition is not None or not self.single_source):
            self.add_retracted()
            self.misinformation = True

    def add_retracted(self) -> None:
        """
        Add retracted belief to random agent.
        :return: None
        """
        if self.single_source:
            a = self.agentZero
        elif self.same_partition is None:
            a = random.choice(self.schedule.agents)
        elif 'partition' in self.G.graph.keys() and self.same_partition:
            if self.delay == 0:
                num = random.choice(list(self.G.graph['partition'][0])[1:])  # retraction source != fake source
            else:
                num = random.choice(list(self.G.graph['partition'][0]))
            a = self.schedule.agents[num]
        elif 'partition' in self.G.graph.keys() and not self.same_partition:
            num = random.choice(list(self.G.graph['partition'][1]))
            a = self.schedule.agents[num]
        else:
            a = random.choice(self.schedule.agents)
        a.set_belief(Belief.Retracted)

    def step(self) -> None:
        """
        Advance the model by one step.
        :return: None
        """
        if (self.delay > 0) and (self.schedule.time == self.delay):
            self.add_retracted()
            self.misinformation = True

        self.schedule.step()

    def logs(self) -> tuple[dict, dict, list[set]]:
        """
        Wrapper for logs function in SimpleLogger class.
        Get agents' belief, interaction and pair history, respectively.
        :return: a tuple containing 3 elements: belief history, interaction history, and pair history
        """
        return self.schedule.logs()
