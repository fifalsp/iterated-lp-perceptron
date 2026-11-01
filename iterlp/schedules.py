"""Margin schedules kappa_1 > kappa_2 > ... used in the experiments."""
from fractions import Fraction


def quadratic_schedule(offset, scale, horizon, rounds):
    """kappa_t = offset + scale * ((horizon - t) / (horizon - 1))**2 for t = 1, ..., rounds.

    Section 8 uses offset = 0.01, scale = 0.5789153654719907, horizon = 150 and
    rounds = 140, so kappa_1 = 0.5889153654719907 and kappa_140 = 0.01260...
    """
    return [offset + scale * ((horizon - t) / (horizon - 1)) ** 2 for t in range(1, rounds + 1)]


def quadratic_schedule_exact(offset, scale, horizon, rounds):
    """The same schedule evaluated in exact rational arithmetic, then rounded.

    Used only to check that saved schedules match the formula to 1e-15.
    """
    return [float(Fraction(str(offset)) + Fraction(str(scale)) * Fraction(horizon - t, horizon - 1) ** 2)
            for t in range(1, rounds + 1)]


def schedule_from_config(config):
    """Schedule described by the "schedule" entry of a configuration file."""
    spec = config['schedule']
    if spec['type'] != 'quadratic':
        raise ValueError('Unknown schedule type: ' + spec['type'])
    return quadratic_schedule(spec['offset'], spec['scale'], spec['horizon'], spec['rounds'])
