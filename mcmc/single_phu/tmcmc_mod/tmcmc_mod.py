"""
@author: Mukesh K. Ramancha
@Modified by: David Clarabut

transitional Markov chain Monte Carlo
a.k.a. sequential Monte Carlo
"""
import numpy as np
import time
# import resampling


def initial_population(N, all_pars):
    """
    Generates initial population from prior distribution

    Parameters
    ----------
    N : float
        number of particles.
    all_pars : list of size Np
        Np is number of parameters
        all_pars[i] is object of type pdfs
        all parameters to be inferred.

    Returns
    -------
    ini_pop : numpy array of size N x Np
        initial population.

    """
    ini_pop = np.zeros((N, len(all_pars)))
    for i in range(len(all_pars)):
        ini_pop[:, i] = all_pars[i].generate_rns(N)
    return ini_pop


def log_prior(s, all_pars):
    """
    computes log_prior value at all particles

    Parameters
    ----------
    s : numpy array of size N x Np
        all particles.
    all_pars : list of size Np
        Np is number of parameters
        all_pars[i] is object of type pdfs
        all parameters to be inferred.

    Returns
    -------
    log_p : numpy array of size N
        log prior at all N particles .

    """
    log_p = 0
    for i in range(len(s)):
        log_p = log_p + all_pars[i].log_pdf_eval(s[i])
    return log_p

# from scipy.special import logsumexp
def compute_beta_update_evidence(beta, log_likelihoods,
                                 log_evidence):
    """
    Computes beta for the next stage and updated model evidence

    Parameters
    ----------
    beta : float
        stage parameter.
    log_likelihoods : numpy array of size N
        log likelihood values at all particles
    log_evidence : float
        log of evidence.
    prev_ESS : int
        effective sample size of previous stage

    Returns
    -------
    new_beta : float
        stage parameter for next stage.
    log_evidence : float
        updated log evidence.
    Wm_n : numpy array of size N
        weights of particles for the next stage
    ESS : float
        effective sample size of new stage

    """
    old_beta = beta
    min_beta = beta
    max_beta = 1.0
    N = len(log_likelihoods)
    # rN = int(len(log_likelihoods) * 0.5)
    # rN = 0.95*prev_ESS
    # rN = max(0.95*prev_ESS, 3000)             # min particles 1000

    while (max_beta - min_beta > 1e-06):       # min step size is 10^-12
        new_beta = 0.5*(max_beta+min_beta)

        # plausible weights of Sm corresponding to new beta
        inc_beta = new_beta-old_beta

        Wm = np.exp(inc_beta*(log_likelihoods - log_likelihoods.max()))

        cov_w = np.std(Wm)/np.mean(Wm)

        if cov_w > 1.0:
            max_beta = new_beta
        else:
            min_beta = new_beta

        Wm_n = Wm/sum(Wm)
        ESS = int(1/np.sum(Wm_n**2))

        # log_Wm = inc_beta * log_likelihoods
        # log_Wm_n = log_Wm - logsumexp(log_Wm)
        # ESS = int(np.exp(-logsumexp(log_Wm_n * 2)))

        # if ESS == rN:
        #     break
        # elif ESS < rN:
        #     max_beta = new_beta
        # else:
        #     min_beta = new_beta

    if np.abs(1.0 - new_beta) < 1e-02:
        new_beta = 1.0

        # plausible weights of Sm corresponding to new beta
        inc_beta = new_beta-old_beta

        Wm = np.exp(inc_beta*(log_likelihoods - log_likelihoods.max()))

        Wm_n = Wm/sum(Wm)
        ESS = int(1/np.sum(Wm_n**2))

        # log_Wm = inc_beta * log_likelihoods
        # log_Wm_n = log_Wm - logsumexp(log_Wm)

    # Wm = np.exp(log_Wm)
    # Wm_n = np.exp(log_Wm_n)

    # update model evidence
    # (check it, might not be correct, as we remove log.likelihood max in compute_beta)
    # evidence = evidence * (sum(Wm)/N)
    # log_evidence = log_evidence + logsumexp(log_Wm) - np.log(N)
    log_evidence = log_evidence + np.log((sum(Wm)/N))

    return new_beta, log_evidence, Wm_n, ESS


def propose(current, covariance, n):
    """
    proposal distribution for MCMC in pertubation stage

    Parameters
    ----------
    current : numpy array of size Np
        current particle location
    covariance : numpy array of size Np x Np
        proposal covariance matrix
    n : int
        number of proposals.

    Returns
    -------
    numpy array of size n x Np
        n proposals.

    """
    return np.random.multivariate_normal(current, covariance, n)


def MCMC_MH(Priormsum, Em, Nm_steps, current, likelihood_current,
            beta, numAccepts, all_pars, log_likelihood):
    """
    Pertubation: Markov chain Monte Carlo using Metropolis-Hastings
    perturbs each particle using MCMC MH

    Parameters
    ----------
    Priormsum : float
        The sum of logprior values
    Em : numpy array of size Np x Np
        proposal covarince matrix.
    Nm_steps : int
        number of perturbation steps.
    current : numpy array of size Np
        current particle location
    likelihood_current : float
        log likelihood value at current particle
    posterior_current : float
        log posterior value at current particle
    beta : float
        stage parameter.
    numAccepts : int
        total number of accepts
    all_pars : : list of size Np
        Np is number of parameters
        all_pars[i] is object of type pdfs
        all parameters to be inferred.
    log_likelihood : function
        log likelihood function to be defined in main.py.

    Returns
    -------
    current : numpy array of size Np
        perturbed particle location
    likelihood_current : float
        log likelihood value at perturbed particle
    posterior_current : float
        log posterior value at perturbed particle
    numAccepts : int
        total number of accepts during perturbation (MCMC - MH)

    """
    all_proposals = []
    all_PLP = []

    deltas = propose(np.zeros(len(current)), Em, Nm_steps)
    for j2 in range(Nm_steps):
        delta = deltas[j2]
        proposal = current+delta
        prior_proposal = log_prior(proposal, all_pars)

        # proposal satisfies the prior constraints
        if np.isfinite(prior_proposal):
            likelihood_proposal = log_likelihood(proposal) - Priormsum
        else:
            likelihood_proposal = -np.Inf   # dont run the FE model

        log_acceptance = beta*(likelihood_proposal - likelihood_current)

        all_proposals.append(proposal)
        all_PLP.append([prior_proposal,
                        likelihood_proposal])

        if np.isfinite(log_acceptance) and (np.log(np.random.uniform())
                                            < log_acceptance):
            # accept
            current = proposal
            likelihood_current = likelihood_proposal
            numAccepts += 1

    # gather all last samples
    return current, likelihood_current, numAccepts


def run_tmcmc(N, all_pars, log_likelihood, parallel_processing,
              status_file_name, Nm_steps_max=5, Nm_steps_maxmax=10):
    """
    main function to run transitional mcmc

    Parameters
    ----------
    N : int
        number of particles to be sampled from posterior
    all_pars : list of size Np
        Np is number of parameters
        all_pars[i] is object of type pdfs
        all parameters to be inferred
    log_likelihood : function
        log likelihood function to be defined in main.py as is problem specific
    parallel_processing : string
        should be either 'multiprocessing' or 'mpi'
    status_file_name : string
        name of the status file to store status of the tmcmc sampling
    Nm_steps_max : int, optional
        Numbers of MCMC steps for pertubation. The default is 5.
    Nm_steps_maxmax : int, optional
        Numbers of MCMC steps for pertubation. The default is 5.

    Returns
    -------
    mytrace: returns trace file of all samples of all tmcmc stages
    comm: if parallel_processing is mpi

    """

    # side note: make all_pars as ordered dict in the future
    # Initialize (beta, effective sample size)
    beta = .0
    ESS = N
    mytrace = []
    stage_num = 0
    start_time_global = time.time()

    # Initialize other TMCMC variables
    Nm_steps = Nm_steps_max
    parallelize_MCMC = True
    Adap_calc_Nsteps = 'yes'    # yes or no
    Adap_scale_cov = 'no'      # yes or no
    scalem = 2.4/np.sqrt(len(all_pars))                  # cov scale factor
    log_evidence = 0            # model evidence
    Chain = np.array([], dtype= np.float64)
    # initial samples
    Sm = initial_population(N, all_pars)

    # Evaluate posterior at Sm
    Priorm = np.array([log_prior(s, all_pars) for s in Sm]).squeeze()

    status_file = open(status_file_name, "a+")

    # Evaluate log-likelihood at current samples Sm
    if parallelize_MCMC:
        iterables = [[Sm[ind]] for ind in range(N)]
        status_file.write('======================== \n')
        print('========================')
        if parallel_processing == 'multiprocessing':
            status_file.write("using multiprocessing \n")
            print("using multiprocessing")
            import multiprocessing as mp
            from multiprocessing import Pool
            pool = Pool(processes=mp.cpu_count())
            Lmt = pool.starmap(log_likelihood, iterables) #This takes the values of iterables and computes the logliklihood function with them.   
        elif parallel_processing == 'mpi':
            status_file.write("using mpi \n")
            print("using mpi")
            # import mpi4py
            # mpi4py.rc.recv_mprobe = False
            from mpi4py import MPI
            from mpi4py.futures import MPIPoolExecutor
            comm = MPI.COMM_WORLD
            executor = MPIPoolExecutor(max_workers=comm.Get_size())
            Lmt = list(executor.starmap(log_likelihood, iterables))
        else:
            raise(AssertionError('parallel_processing invalid, should be either multiprocessing or mpi'))
        status_file.write('======================== \n')
        print('========================')
        #Lm = np.array(Lmt).squeeze() #an array of all liklihood function values from their priors.
    else:
        Lmt = [log_likelihood(Sm[j1]) for j1 in range(N)] #cases when parallel processing is not used.
        #Lm = np.array([log_likelihood(ind, Sm[ind])
                       #for ind in range(N)]).squeeze()
    Lm = np.array(Lmt).squeeze() #conversion to an array
    Priormsum = np.sum(Priorm)
    for j3 in range(N):
        Lm[j3] = -Priormsum + Lm[j3]

    while beta < 1:
        stage_num += 1
        start_time_stage = time.time()

        # adaptivly compute beta s.t. ESS = N/2 or ESS = 0.95*prev_ESS
        # plausible weights of Sm corresponding to new beta
        beta, log_evidence, Wm_n, ESS = compute_beta_update_evidence(beta, Lm, log_evidence)

        # Calculate covaraince matrix using Wm_n
        Cm = np.cov(Sm, aweights=Wm_n, rowvar=0)

        # Resample ###################################################
        # Resampling using plausible weights
        SmcapIDs = np.random.choice(range(N), N, p=Wm_n)
        # SmcapIDs = resampling.stratified_resample(Wm_n)
        Smcap = Sm[SmcapIDs] #positions
        Lmcap = Lm[SmcapIDs] #values of the likelihood

        # save to trace
        # stage m: samples, likelihood, weights, next stage ESS, next stage beta, resampled samples
        mytrace.append([Sm, Lm, Wm_n, ESS, beta, Smcap])

        # print to status_file
        status_file = open(status_file_name, "a+")
        status_file.write("stage number = %d \n" % stage_num)
        print("stage number = %d" % stage_num)
        status_file.write("beta = %.5f \n" % np.float64(beta))
        print("beta = %.5f" % np.float64(beta))
        status_file.write("ESS = %d \n" % ESS)
        print("ESS = %d" % ESS)
        status_file.write("scalem = %.2f \n" % scalem)
        print("scalem = %.2f" % scalem)

        # Perturb ###################################################
        # perform MCMC starting at each Smcap (total: N) for Nm_steps
        Em = ((scalem)**2) * Cm     # Proposal dist covariance matrix

        numProposals = N*Nm_steps
        numAccepts = 0

        if parallelize_MCMC:
            iterables = [(Priormsum,Em, Nm_steps, Smcap[j1], Lmcap[j1],
                          beta, numAccepts, all_pars,
                          log_likelihood) for j1 in range(N)] #put all function inputs onto an iterables variable for parallel processing below.
            if parallel_processing == 'multiprocessing':
                results = pool.starmap(MCMC_MH, iterables)
            elif parallel_processing == 'mpi':
                results = list(executor.starmap(MCMC_MH, iterables))
        else:
            results = [MCMC_MH(Priormsum,Em, Nm_steps, Smcap[j1], Lmcap[j1],
                               beta, numAccepts, all_pars,
                               log_likelihood) for j1 in range(N)] #cases when parallel processing is not used.

        Sm1, Lm1, numAcceptsS = zip(*results) #extract the results from parallel processing.
        Sm1 = np.asarray(Sm1) #conversion to an asarray.
        Lm1 = np.asarray(Lm1)
        numAcceptsS = np.asarray(numAcceptsS)
        numAccepts = sum(numAcceptsS)

        # total observed acceptance rate
        R = numAccepts/numProposals
        status_file.write("acceptance rate = %.2f \n" % R)
        print("acceptance rate = %.2f" % R)

        # Calculate Nm_steps based on observed acceptance rate
        if Adap_calc_Nsteps == 'yes':
            # increase max Nmcmc with stage number
            Nm_steps_max = min(Nm_steps_max+2, Nm_steps_maxmax)
            status_file.write("adapted max MCMC steps = %d \n" % Nm_steps_max)
            print("adapted max MCMC steps = %d" % Nm_steps_max)

            acc_rate = max(1. / numProposals, R)
            Nm_steps = min(Nm_steps_max, 1 + int(np.log(1-0.99)
                                                 / np.log(1 - acc_rate)))
            status_file.write("next MCMC Nsteps = %d \n" % Nm_steps)
            print("next MCMC Nsteps = %d" % Nm_steps)

        status_file.write("log_evidence till now = %.20f \n" % log_evidence)
        print("log_evidence till now = %.20f" % log_evidence)
        status_file.write("--- Execution time: %.2f mins --- \n" %
                          ((time.time() - start_time_stage)/60))
        print("--- Execution time: %.2f mins ---" %
                          ((time.time() - start_time_stage)/60))
        status_file.write('======================== \n')
        print('========================')
        status_file.close()

        # scale factor based on observed acceptance ratio
        if Adap_scale_cov == 'yes':
            scalem = (1/9)+((8/9)*R)

        # append current samples to the chain:
        Chain = np.append(Chain, Sm1)
        # for next beta
        Sm, Lm = Sm1, Lm1

    # save to trace
    mytrace.append([Sm, Lm, np.ones(len(Wm_n))/len(Wm_n),
                    'notValid', 1, 'notValid'])

    status_file = open(status_file_name, "a+")
    status_file.write("--- Execution time: %.2f mins --- \n" %
                      ((time.time() - start_time_global)/60))
    print("--- Execution time: %.2f mins ---" %
                      ((time.time() - start_time_global)/60))
    status_file.write("log_evidence = %.20f \n" % log_evidence)
    print("log_evidence = %.20f" % log_evidence)

    if parallelize_MCMC:
        if parallel_processing == 'multiprocessing':
            status_file.write("closing multiprocessing \n")
            print("closing multiprocessing")
            pool.close()
        elif parallel_processing == 'mpi':
            status_file.write("shutting down mpi \n")
            print("shutting down mpi")
            executor.shutdown()

    status_file.close()

    if parallel_processing == 'multiprocessing':
        return Sm, Chain, mytrace, None
    elif parallel_processing == 'mpi':
        return Sm, Chain, mytrace, comm
