"""Test examples provided by nimfa"""

import numpy as np

from nimfa.examples import synthetic

import pytest

synthetic_runs = [getattr(synthetic, f) for f in dir(synthetic) if f.startswith('run_')]

@pytest.mark.medium
@pytest.mark.parametrize('synth_run', synthetic_runs)
def test_synthetic(synth_run):
    def check_synthetic_result(fit, idx=None):
        """Basic checks that synthetic results are 'Ok'"""
        # typically tests pass with > .75 but because of still present inherent
        # randomizations results vary. So to be on a safe side for now set a very
        # liberal threshold
        # TODO:  make runs deterministic
        assert fit.summary(idx)['evar'] > 0.62

    # Setup data exactly the same way as in synthetic example
    prng = np.random.RandomState(42)
    # construct target matrices
    V = prng.rand(20, 30)
    V1 = prng.rand(20, 25)

    try:
        old_print_info = synthetic.print_info
        synthetic.print_info = check_synthetic_result
        if synth_run is getattr(synthetic, 'run_snmnmf'):
            synth_run(V, V1)
        else:
            synth_run(V)
    finally:
        synthetic.print_info = old_print_info