from __future__ import annotations
from typing import TYPE_CHECKING
from mesa import Agent
from belief import Belief, Mode
import random


if TYPE_CHECKING:
    # To prevent cyclic dependency...
    from model import AcademicLiterature


class Article(Agent):
    """
    An Agent with some initial information.
    """
    def __init__(self, unique_id: int, model: AcademicLiterature, cited_by: list, share_time: float):
        """
        Initialize an Article.
        :param unique_id: an integer unique to a single agent within the predefined model
        :param model: the model in which the agent will exist
        :param cited_by: a list of neighboring agents represented as nodes in a NetworkX graph
        :param share_time: a float indicating the time limit within which new beliefs are shared; np.inf if endless
        """
        super().__init__(unique_id, model)

        # Default params
        self.belief = Belief.Neutral
        self.cited_by = cited_by
        self.clock = 0  # internal timer (absolute time)
        self.belief_time = 0  # time current belief has been held
        self.share_time = share_time
        self.delay = model.delay  # time to delay before introducing retraction

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

    def update(self, other: Article) -> None:
        """
        Update agent's own beliefs based on its interaction with another agent.
        :param other: another Article
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
            # if original info has been retracted, 5.4% chance false belief is not passed on
            # https://direct.mit.edu/qss/article/2/4/1144/107356/Continued-use-of-retracted-papers-Temporal-trends
            if self.clock >= self.delay:
                if random.randint(0, 1000) > 54:
                    self.set_belief(Belief.Fake)
            else:
                self.set_belief(Belief.Fake)

        # Convert self to retracted belief
        if self.belief == Belief.Fake and other.belief == Belief.Retracted and is_sharing_retracted:
            self.set_belief(Belief.Retracted)

    def step(self, interlocutor: Article) -> None:
        """
        Overloads Agent's step method. Wrapper for Article's update method.
        Update agent's own beliefs based on its interaction with another agent.
        :param interlocutor: another Article
        :return: None
        """
        self.update(interlocutor)
