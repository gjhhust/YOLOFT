# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.utils import SETTINGS

try:
    import ray
    from ray import tune
    from ray.air import session

    assert SETTINGS['raytune'] is True  # verify integration is enabled
except (ImportError, AssertionError):
    tune = None


def on_fit_epoch_end(trainer):
    """Sends training metrics to Ray Tune at end of each epoch."""
    if ray.train._internal.session._get_session():
        metrics = trainer.metrics
        metrics['epoch'] = trainer.epoch
        ray.train._internal.session.report(metrics)


callbacks = {
    'on_fit_epoch_end': on_fit_epoch_end, } if tune else {}
