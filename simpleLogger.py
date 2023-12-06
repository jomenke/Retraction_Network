from __future__ import annotations
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    # To prevent cyclic dependency...
    from agent import Article
    from model import AcademicLiterature


class SimpleLogger:
    """A simple logger tracking agents' belief and interaction history."""
    def __init__(self, model: AcademicLiterature):
        """
        Initialize a SimpleLogger.
        :param model: a AcademicLiterature to be logged
        """
        self.model = model
        self.belief_history = dict()  # unique_id: belief.value
        self.interaction_history = dict()  # unique_id: [(interlocutor_id, interlocutor.belief)]
        self.pair_history = []   # [{agentA, agentB}, {agentA, agentB}]

    def add(self, agent: Article) -> None:
        """
        Add agent to logger.
        :param agent: an Article instance to add to logger
        :return: None
        """
        self.belief_history[agent.unique_id] = [agent.belief.value]
        self.interaction_history[agent.unique_id] = []

    def logs(self) -> tuple[dict, dict, list[set]]:
        """
        Get agents' belief, interaction and pair history, respectively.
        :return: a tuple containing 3 elements: belief history, interaction history, and pair history
        """
        return self.belief_history, self.interaction_history, self.pair_history

    def log(self, agent_a: Article, agent_b: Article) -> None:
        """
        Log agents to belief and interaction history; agent order does not matter.
        :param agent_a: an interacting Article
        :param agent_b: an interacting Article
        :return: None
        """
        agents = self.model.schedule.agents

        # log belief history
        for agent in agents:
            self.belief_history[agent.unique_id].append(agent.belief.value)

        # log interaction history
        entry_a = (agent_a.unique_id, agent_a.belief)
        entry_b = (agent_b.unique_id, agent_b.belief)
        self.interaction_history[agent_a.unique_id].append(entry_b)
        self.interaction_history[agent_b.unique_id].append(entry_a)

        # log pair history
        pair = set((agent_a.unique_id, agent_b.unique_id))  # a set of set(id pairs)
        self.pair_history.append(pair)
