from __future__ import annotations
from typing import TYPE_CHECKING
from mesa import Agent
from belief import Belief, Mode


if TYPE_CHECKING:
    # To prevent cyclic dependency...
    from model import KnowledgeModel


class PopAgent(Agent):
    """An Agent with some initial knowledge."""
    def __init__(self, unique_id: int, model: KnowledgeModel, neighbors: list, share_time: float):
        """
        Initialize a PopAgent.
        :param unique_id: an integer unique to a single agent within the predefined model
        :param model: the model in which the agent will exist
        :param neighbors: a list of neighboring agents represented as nodes in a NetworkX graph
        :param share_time: a float indicating the time limit within which new beliefs are shared; np.inf if endless
        """
        super().__init__(unique_id, model)

        # Default params
        self.belief = Belief.Neutral
        self.neighbors = neighbors
        self.clock = 0  # internal timer (absolute time)
        self.belief_time = 0  # time current belief has been held
        self.share_time = share_time

    def tick(self) -> None:
        """
        Increment clock by 1.
        :return: None
        """
        self.clock += 1
        self.belief_time += 1

    def is_sharing(self) -> bool:
        """
        Check if agent is still sharing own belief.
        :return: bool; if True, then agent is able to share belief
        """
        return self.belief_time <= self.share_time

    def set_belief(self, belief: Belief) -> None:
        """
        Set agent's belief.
        :param belief: predefined in Belief class; integer mapped to a specific belief (e.g., neutral = 0)
        :return: None
        """
        self.belief = belief
        self.belief_time = 0

    def update(self, other: PopAgent) -> None:
        """
        Update agent's own beliefs based on its interaction with another agent.
        :param other: another PopAgent
        :return: None
        """

        # Check if other is sharing belief (model dependent)
        if self.model.mode == Mode.TimedNovelty:
            is_sharing_fake = other.is_sharing()
            is_sharing_retracted = other.is_sharing()
        elif self.model.mode == Mode.CorrectionFatigue:
            is_sharing_fake = True
            is_sharing_retracted = other.is_sharing()
        else:
            is_sharing_fake = True
            is_sharing_retracted = True

        # Convert self to false belief
        if self.belief == Belief.Neutral and other.belief == Belief.Fake and is_sharing_fake:
            self.set_belief(Belief.Fake)

        # Convert self to retracted belief
        if self.belief == Belief.Fake and other.belief == Belief.Retracted and is_sharing_retracted:
            self.set_belief(Belief.Retracted)

    def step(self, interlocutor: PopAgent) -> None:
        """
        Overloads Agent's step method. Wrapper for PopAgent's update method.
        Update agent's own beliefs based on its interaction with another agent.
        :param interlocutor: another PopAgent
        :return: None
        """
        self.update(interlocutor)
