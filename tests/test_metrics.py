"""Tests for grpo_math.trainer.metrics -- the line-buffered JSONL metrics
logger used by the training loop.
"""

from grpo_math.trainer.metrics import MetricsLogger, read_metrics


def test_log_read_roundtrip(tmp_path):
    path = tmp_path / "metrics.jsonl"
    rows = [
        {"step": 0, "loss": 1.5, "kl": 0.01},
        {"step": 1, "loss": 1.25, "nested": {"a": 1, "b": [1, 2, 3]}},
        {"step": 2, "loss": 0.9999, "flag": True, "note": None},
    ]
    with MetricsLogger(path) as logger:
        for row in rows:
            logger.log(row)

    assert read_metrics(path) == rows


def test_append_on_resume(tmp_path):
    path = tmp_path / "metrics.jsonl"

    logger = MetricsLogger(path)
    logger.log({"step": 0})
    logger.log({"step": 1})
    logger.close()

    logger = MetricsLogger(path, resume=True)
    logger.log({"step": 2})
    logger.close()

    rows = read_metrics(path)
    assert rows == [{"step": 0}, {"step": 1}, {"step": 2}]


def test_truncate_without_resume(tmp_path):
    path = tmp_path / "metrics.jsonl"

    logger = MetricsLogger(path)
    logger.log({"step": 0})
    logger.log({"step": 1})
    logger.close()

    logger = MetricsLogger(path, resume=False)
    logger.log({"step": 99})
    logger.close()

    rows = read_metrics(path)
    assert rows == [{"step": 99}]


def test_flush_per_call(tmp_path):
    path = tmp_path / "metrics.jsonl"

    logger = MetricsLogger(path)
    try:
        logger.log({"step": 0})
        # No close() yet -- a concurrent reader (e.g. `tail -f` on a running
        # pod) must see the row immediately because log() flushes per call.
        assert read_metrics(path) == [{"step": 0}]

        logger.log({"step": 1})
        assert read_metrics(path) == [{"step": 0}, {"step": 1}]
    finally:
        logger.close()
