import networkx as nx
import numpy as np

from qaoa_experiment.simulation import optimize

def test_fix_param_gauge(verbose=False):
    """
    Test to show optimize.fix_param_gauge() can fully reduce redundancies of QAOA parameters
    """

    tolerance = 1e-10

    # original parameters (at p=3) in preferred gauge
    param = np.array([0.2, 0.4, 0.7, -0.6, -0.5, -0.3])

    # copy of parameters
    param2 = param.copy()

    # test that gammas are periodic in pi
    param2[:3] += np.random.choice([1,-1],3)*np.pi
    param_fixed = optimize.fix_param_gauge(param2)
    assert np.linalg.norm(param-param_fixed) <= tolerance

    # test that betas are periodic in pi/2
    param2[3:] += np.random.choice([1,-1],3)*np.pi/2
    param_fixed = optimize.fix_param_gauge(param2)
    assert np.linalg.norm(param-param_fixed) <= tolerance

    ## Case: ODD_DEGREE_ONLY
    # test that shifting gamma_i by (n+1/2)*pi and beta_{j>=i} -> -beta_{j>=i}
    # gives equivalent parameters
    param2[2] -= np.pi/2
    param2[5] = -param2[5]
    param_fixed = optimize.fix_param_gauge(param2, degree_parity=optimize.ODD_DEGREE_ONLY)
    assert np.linalg.norm(param-param_fixed) <= tolerance

    param2[1] += np.pi*3/2
    param2[4:] = -param2[4:]
    param_fixed = optimize.fix_param_gauge(param2, degree_parity=optimize.ODD_DEGREE_ONLY)
    assert np.linalg.norm(param-param_fixed) <= tolerance

    # test that two inequivalent parameters should not be the same after fixing gauge
    param2 = param.copy()
    param2[0] -= np.pi*3/2
    param_fixed = optimize.fix_param_gauge(param2, degree_parity=optimize.ODD_DEGREE_ONLY)
    assert np.linalg.norm(param-param_fixed) > tolerance

    ## Case: EVEN_DEGREE_ONLY - test parameters are periodic in pi/2
    param2 = param.copy()
    param2 += np.random.choice([1,-1],6)*np.pi/2
    param_fixed = optimize.fix_param_gauge(param2, degree_parity=optimize.EVEN_DEGREE_ONLY)
    assert np.linalg.norm(param-param_fixed) <= tolerance
