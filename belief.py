"""
This module defines the beliefs and mode to be utilized within the simulation.
"""

from enum import Enum


class Belief(Enum):
    """An encoding of all possible beliefs."""
    Neutral = 0
    Fake = 1
    Retracted = 2


class Mode(Enum):
    """An encoding of different modes."""
    Default = 0
    TimedNovelty = 1
    CorrectionFatigue = 2
