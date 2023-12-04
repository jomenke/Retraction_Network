from mesa import Agent
from mesa import Model


class BaseScheduler:
    """
    Simplest scheduler; activates agents one at a time, in the order they were added.
    Assumes that each agent added has a *step* method which takes no arguments.
    (This is explicitly meant to replicate the scheduler in MASON).
    """
    def __init__(self, model: Model):
        """
        Initialize a new, empty BaseScheduler.
        :param model: a simulation Model
        """
        self.model = model
        self.steps = 0
        self.time = 0
        self.agents = []

    def add(self, agent: Agent) -> None:
        """
        Add an Agent object to the schedule.
        :param agent: An Agent to be added to the schedule. NOTE: The agent must have a step() method.
        :return: None
        """
        self.agents.append(agent)

    def remove(self, agent: Agent) -> None:
        """
        Remove all instances of a given agent from the schedule.
        :param agent: An agent object.
        :return: None
        """
        while agent in self.agents:
            self.agents.remove(agent)

    def step(self) -> None:
        """
        Execute the step of all the agents, one at a time.
        :return: None
        """
        for agent in self.agents[:]:
            agent.step()
        self.steps += 1
        self.time += 1

    def get_agent_count(self) -> int:
        """
        Get the current number of agents in the queue.
        :return: integer; number of queued agents
        """
        return len(self.agents)

# class SingleRandomActivation(BaseScheduler):
#
#     def step(self) -> None:
#         """
#         Executes the step of all agents, one at a time, in random order.
#         :return: None
#         """
#         random.shuffle(self.agents)
#         for agent in self.agents[:1]:
#             agent.step()
#         self.steps += 1
#         self.time += 1
