"""Test examples provided by nimfa"""

import numpy as np

from nimfa.examples import synthetic
from nimfa.examples import orl_images
from nimfa.examples import all_aml
from nimfa.examples import medulloblastoma
from nimfa.examples import recommendations
from nimfa.examples import cbcl_images


import pytest

synthetic_runs = [getattr(synthetic, f) for f in dir(synthetic) if f.startswith('run_')]

@pytest.mark.medium
@pytest.mark.parametrize('synth_run', synthetic_runs)
def test_synthetic(synth_run, monkeypatch):
    ran_funcs = []
    def check_synthetic_result(fit, idx=None):
        """Basic checks that synthetic results are 'Ok'"""
        ran_funcs.append(1)
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

    monkeypatch.setattr(synthetic, 'print_info', check_synthetic_result)
    if synth_run is getattr(synthetic, 'run_snmnmf'):
        synth_run(V, V1)
    else:
        synth_run(V)
    # for paranoid monkeypatching people, to make sure that the patch worked
    assert ran_funcs != [], "we must have ran our check"

#Check if the example datasets run
@pytest.mark.slow
def test_orl():
    orl_images.run()

@pytest.mark.slow
def test_aml():
    all_aml.run()

@pytest.mark.slow
def test_medulloblastoma():
    medulloblastoma.run()    

#@pytest.mark.slow
#def test_recommendations():
#    recommendations.run()    

@pytest.mark.slow
def test_cbcl():
    cbcl_images.run()    
