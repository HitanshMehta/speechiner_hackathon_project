"""Evaluate extractors that fill a form from a live transcript.

What a submission needs::

    from live_form_eval import Extractor, Session

    class MyExtractor(Extractor):
        async def run(self, session: Session) -> None:
            async for partial in session.partials():
                ...
                session.update_snapshot({...})
"""

from live_form_eval.config import Config, load_config
from live_form_eval.interface.extractor import Extractor
from live_form_eval.interface.schema import FieldSpec, FieldType, FieldValue, Form, OptionsConfig, RangeConfig
from live_form_eval.interface.session import InvalidValue, Session
from live_form_eval.interface.transcript import Partial, Word

__all__ = ["Config", "Extractor", "FieldSpec", "FieldType", "FieldValue", "Form", "InvalidValue", "OptionsConfig",
           "Partial", "RangeConfig", "Session", "Word", "evaluate", "load_config", "run"]


def run(*args, **kwargs):
    """Run an extractor on a dataset and write a run folder. See :func:`live_form_eval.harness.runner.run`."""
    from live_form_eval.harness.runner import run as _run

    return _run(*args, **kwargs)


def evaluate(*args, **kwargs):
    """Score run folders against the gold. See :func:`live_form_eval.evaluation.evaluate.evaluate`."""
    from live_form_eval.evaluation.evaluate import evaluate as _evaluate

    return _evaluate(*args, **kwargs)
