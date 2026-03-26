from .router import route
from .evaluator import evaluate
from .ranker import rank
from .worker import work

__all__ = ["route", "evaluate", "rank", "work", "run_pipeline"]


def run_pipeline(email_batch):
    pass
