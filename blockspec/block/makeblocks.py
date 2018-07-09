# This algorithm is a copy from Robert Lauers lctoos with small adaptions
# https://github.com/rjlauer/lctools/blob/master/lctools/makeblocks.py

import numpy as np


######################################################################
# Baysian Blocks
######################################################################


def makeblocks(smjd, srate, srate_err, ncp_prior):
    """
    Takes the times in MJD, smjd the datapoints srate, its error srate_error and
    the choice of the prior npc_prior as input.
    It calculates the bayesian blocks and returns
    s_bbpoints, s_bbtimes, s_bbamps, s_bbampserr, s_cp, s_amplitudes, s_amplitudes_err
    """

    N = len(srate)

    best = []
    last = []

    if N > 15000:

        for R in range(1, N+1):
            sum_a = 1/2.*np.array([np.sum (1./srate_err[k:R]**2) for k in range(0, R)])
            sum_b = -np.array([np.sum(srate[k:R]/srate_err[k:R]**2) for k in range(0, R)])
            fit_vec = sum_b**2./(4.*sum_a)
            fullFitVec = np.concatenate([[0.], best]) + fit_vec - ncp_prior
            last.append(np.argmax(fullFitVec))
            best.append(fullFitVec[last[R-1]])

    else:

        a = np.triu(np.ones([N, N]), k=0)
        b = np.tile((srate / (srate_err ** 2)), (N, 1)) * a
        a *= np.tile(1. / (srate_err ** 2), (N, 1))

        for R in range(1, N + 1):
            sum_a = 2. * np.sum(a[:R, :R], axis=1)
            sum_b = np.sum(b[:R, :R], axis=1)**2
            fit_vec = sum_b / (sum_a)
            last.append(np.argmax(np.concatenate([[0.], best]) + fit_vec - ncp_prior))
            best.append((np.concatenate([[0.], best]) + fit_vec - ncp_prior)[last[R - 1]])

    s_index = last[N - 1]
    s_ncp = 0
    # one entry per change point:
    s_bbpoints = np.array([], dtype=int)
    s_bbamps = np.array([], dtype=float)
    s_bbampserr = np.array([], dtype=float)
    smjd = np.array(smjd)
    s_bbtimes = np.array([], dtype=(smjd[1]-smjd[-1]).dtype)
    # two entries per change point for start and stop:
    s_cp = np.array([N - 1], dtype=int)
    s_amplitudes = np.array([], dtype=float)
    s_amplitudes_err = np.array([], dtype=float)
    while (s_index > 0):
        s_bbpoints = np.concatenate([[s_index], s_bbpoints])
        s_cp = np.concatenate([[s_index, s_index], s_cp])
        # amplitudes
        sum_a = 1 / 2. * np.sum(1. / srate_err[s_index:s_cp[-(1 + 2 * s_ncp)]] ** 2.)
        sum_b = -np.sum(srate[s_index:s_cp[-(1 + 2 * s_ncp)]] / srate_err[s_index:s_cp[-(1 + 2 * s_ncp)]] ** 2.)
        amp = -sum_b / (2. * sum_a)
        s_bbamps = np.concatenate([[amp], s_bbamps])
        s_amplitudes = np.concatenate([[amp, amp], s_amplitudes])
        # errors from error on average: sqrt( sum (1/sigma**2) )
        aerr = 1. / np.sqrt(2. * sum_a)
        s_bbampserr = np.concatenate([[aerr], s_bbampserr])
        s_amplitudes_err = np.concatenate([[aerr, aerr], s_amplitudes_err])

        if (s_ncp == 0):
            s_bbtimes = np.concatenate([[smjd[N - 1] - smjd[s_index]], s_bbtimes])
        else:
            s_bbtimes = np.concatenate([[smjd[s_bbpoints[1]] - smjd[s_index]], s_bbtimes])
        s_ncp += 1
        s_index = last[s_index - 1]
    s_bbpoints = np.concatenate([[0], s_bbpoints])
    s_cp = np.concatenate([[0], s_cp])
    sum_a = 1 / 2. * np.sum(1. / srate_err[0:s_cp[1]] ** 2.)
    sum_b = -np.sum(srate[0:s_cp[1]] / srate_err[0:s_cp[1]] ** 2.)
    firstamp = -sum_b / (2. * sum_a)
    s_amplitudes = np.concatenate([[firstamp, firstamp], s_amplitudes])
    s_bbamps = np.concatenate([[firstamp], s_bbamps])
    firstamp_err = 1. / np.sqrt(2. * sum_a)
    s_amplitudes_err = np.concatenate([[firstamp_err, firstamp_err], s_amplitudes_err])
    s_bbampserr = np.concatenate([[firstamp_err], s_bbampserr])
    s_bbtimes = np.concatenate([[smjd[s_cp[1]] - smjd[0]], s_bbtimes])
    # RMS
    # s_bbRMS.append(sqrt(1/(len(srate[s_index:s_cp[-(1+2*s_ncp)]])-1)*np.sum(srate[s_index:s_cp[-(1+2*s_ncp)]]**2.)))

    s_bbamps = np.array(s_bbamps)
    s_bbampserr = np.array(s_bbampserr)
    s_bbtimes = np.array(s_bbtimes)
    s_cp = np.array(s_cp)
    return (s_bbpoints, s_bbtimes, s_bbamps, s_bbampserr, s_cp, s_amplitudes, s_amplitudes_err)
