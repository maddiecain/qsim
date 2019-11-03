import numpy as np



EVEN_DEGREE_ONLY, ODD_DEGREE_ONLY = 0, 1
# TODO: Fix the dependence on the parameter space function on the specific parameters
# Perhaps make them take in the parameters and periods
# Write a separate function to determine the beta period from the parity


def fix_param_gauge(param, gamma_period=np.pi, beta_period=np.pi/2, degree_parity=None):
    """ Use symmetries to reduce redundancies in the parameter space
    This is useful for the interp heuristic that relies on smoothness of parameters

    Based on arXiv:1812.01041 and https://github.com/leologist/GenQAOA/
    """
    p = len(param) // 2

    gammas = np.array(param[:p]) / gamma_period
    betas = -np.array(param[p:2*p]) / beta_period
    # We expect gamma to be positive and beta to be negative, so flip sign of beta for now and flip it back later

    # Reduce the parameters to be between [0, 1] * period
    gammas = gammas % 1
    betas = betas % 1

    # Use time-reversal symmetry to make first gamma small
    if (gammas[0] > 0.25 and gammas[0] < 0.5) or gammas[0] > 0.75:
        gammas = -gammas % 1
        betas = -betas % 1

    # Further simplification if all nodes have same degree parity
    if degree_parity == EVEN_DEGREE_ONLY: # Every node has even degree
        gamma_period = np.pi/2
        gammas = (gammas * 2) % 1
    elif degree_parity == ODD_DEGREE_ONLY: # Every node has odd degree
        for i in range(p):
            if gammas[i] > 0.5:
                gammas[i] = gammas[i] - 0.5
                betas[i:] = 1-betas[i:]

    for i in range(1,p):
        # try to impose smoothness of gammas
        delta = gammas[i] - gammas[i-1]
        if delta >= 0.5:
            gammas[i] -= 1
        elif delta <= -0.5:
            gammas[i] += 1

        #  Try to impose smoothness of betas
        delta = betas[i] - betas[i-1]
        if delta >= 0.5:
            betas[i] -= 1
        elif delta <= -0.5:
            betas[i] += 1

    return np.concatenate((gammas*gamma_period, -betas*beta_period, param[2*p:]*np.pi)).tolist()
