"""Task registry and data pipeline.

    from decider import data as D
    D.Example, D.Q                      one context + typed questions (option list, gold index)
    D.TASKS, D.load_task, D.load_all    ~95 public decision datasets behind one format (tasks_*.py register themselves)
    D.load_cache(path)                  a pickled (train examples, {task: eval examples}) pair

    python -m decider.data.core                 download + convert every task    -> data/tasks.pkl
    python -m decider.data.teacher_labels ...   label descriptions from a local teacher model
    python -m decider.data.teacher_questions .. teacher-written custom questions / routing messages
    python -m decider.data.mixture              the training mixture + probes    -> data/mixture.pkl, data/probes.pkl
"""
from .core import *                                                                  # noqa: F401,F403
from .core import Q, Example, TASKS, task, load_cache, load_task, load_all, TRAIN_CAP, EVAL_CAP, SEED      # noqa: F401
from . import tasks_extended, tasks_agents, tasks_heldout                            # noqa: F401  (register their tasks)
