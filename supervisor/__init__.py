"""Supervisor for OpenHands coding agents running on local models.

OpenHands does the work. The supervisor does what a human babysitter does:
feeds one milestone at a time, verifies "done" with a real command, answers
"should I continue?", and restarts a milestone in a fresh conversation when
the agent gets lost.
"""

__version__ = "0.1.0"
