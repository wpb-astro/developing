"""
rewriting the flux completeness curve fitting

author: Will Bowman
date: 2020-10-15
-------------------------------------------------------------------------------

>>> Description:

Fit the (linear) flux distribution according to the modified Fleming curve
described in Bowman et al. 2019, ApJ, 875, 152

* Assume underlying flux distribution follows a power law in the regime where
  incompleteness begins to matter
* Fleming function describes completeness as a function of flux
* Probability distribution is the product of the underlying power law
  and the completeness fraction
* Introduce modification to the Fleming curve: force the solution to 
  approach zero as the flux approaches zero (i.e., counteract the power law
  which approaches infinity as flux approaches zero)
* Modified Fleming curve only differs from the original Fleming function
  at very low completeness fraction
* Final probability distribution is parameterized by three variables:
      . beta  : power law index of the underlying power law distribution
      . f_50  : the (linear) flux at which completeness = 50%
      . alpha : describes how sharply the completeness fraction declines

>>> Equations:

(1) Original Fleming function

                             1                 alpha * log( f / f_50 )
       F_F(f; f_50, alpha) = - * [ 1 + -------------------------------------- ]
                             2          sqrt( 1 + (alpha * log(f/f_50))**2 )

(2) Exponential decay (modification at low fluxes)

       tau(f) = 1 - exp( - f / f_20 )

       where f_20 is the flux at which the completeness, F_F(f), reaches 20%
             (i.e., it is not an additional model parameter) 

(3) Modified Fleming function

       F_C(f) = [ F_F(f) ] ** ( 1 / tau(f) )

(4) The observed flux probability distribution

       p(f; beta, f_50, alpha) = f**(-beta) * F_C(f) 

>>> Usage:

* read in the linear flux array (in units 1e-17 ergs / s / cm2)
* fit the bright end with a power law:
  . examine grism spectra, decide the flux regime where completeness ~ 100%
  . explore further: use "inspect_powerlaw_index" function
  . restrict the prior on power law index, beta (see "get_param_ranges" function)
* fit the full model: use "fit_model" function
  . can fit a single field, or multiple fields at once
    (force beta, alpha to be the same across all fields)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import emcee 
import time
import corner

import seaborn as sns
sns.set_context("talk") # options include: talk, poster, paper
sns.set_style("ticks")
sns.set_style({"xtick.direction": "in","ytick.direction": "in",
               "xtick.top":True, "ytick.right":True,
               "xtick.major.size":12, "xtick.minor.size":4,
               "ytick.major.size":12, "ytick.minor.size":4,
               })
### color palettes
colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
colors += ["cloudy blue", "browny orange", "dark sea green"]
sns.set_palette(sns.xkcd_palette(colors))
orig_palette = sns.color_palette()


def get_param_ranges(fit_powerlaw_index=False):
    '''
    Set the possible range of values for the model parameters

    Parameters
    ----------
    fit_powerlaw_index : bool
        if True, fitting only the power law index at the bright end
                 used to constrain the power law index when fitting full model
        else, fitting the full three-parameter model

    Model Parameters (i.e., not parameters of this function)
    ----------------
    beta  : the power law index of the underlying flux distribution
    f50   : linear flux at which the completeness fraction is 50%
    alpha : describes the speed with which the completeness declines

    Returns
    -------
    if fitting only the powerlaw index:
        1 list : [min, max] values for the power law index
        --> used to inform the ranges when fitting the full model
    if fitting the true model:
        3 lists : [min, max] for each of the three parameters
                  [beta_lo, beta_hi], [f50_lo, f50_hi], [alpha_lo, alpha_hi]
    '''
    if fit_powerlaw_index:
        betas = [2,5]
        return betas

    else:
        betas  = [3.0,3.2] 
        f50s   = [2.5, 7.5]
        alphas = [2,6] 
        return betas, f50s, alphas

def pf_powerlaw_norm(f, beta, fluxmin, fluxmax=40):
    '''
    Normalized probability distribution for the underlying power law

    Parameters
    ----------
    f : float
        linear flux
    beta : float
        the power law index of the underlying flux distribution
    fluxmin : float
        minimum flux to fit the power law
    fluxmax : float
        maximum flux to fit the power law 

    Returns
    -------
    p : float
        the probability of finding an object with flux f
    '''
    # only consider fluxes in the range of interest
    sel = (f>=fluxmin) & (f<=fluxmax)
    f=f[sel]

    # normalize s.t. probability under the curve = 1
    def powerlaw_nonnorm(f, beta):
        return f ** -beta

    I = quad(powerlaw_nonnorm, fluxmin, fluxmax, args=(beta))[0]
    N = 1. / I
    p = N * powerlaw_nonnorm(f, beta)
    return p

def lnlike_powerlaw(beta, f, fluxmin, fluxmax):
    '''
    log likelihood for the underlying power law flux distribution

    Parameters
    ----------
    beta : float
        the power law index of the underlying flux distribution
    f : 1d array
        linear fluxes
    fluxmin : float
        minimum flux to fit the power law
    fluxmax : float
        maximum flux to fit the power law 

    Returns
    -------
    lnlike : float
        log likelihood 
    '''
    pf = pf_powerlaw_norm(f, beta, fluxmin, fluxmax)
    return sum( np.log(pf[pf!=0]) )

def lnprob_powerlaw(beta, f, fluxmin, fluxmax):
    '''
    log probability of the power law flux distribution

    Parameters
    ----------
    beta : float
        the power law index of the underlying flux distribution
    f : 1d array
        linear fluxes
    fluxmin : float
        minimum flux to fit the power law
    fluxmax : float
        maximum flux to fit the power law 

    Returns
    -------
    log prior + log likelihood

    '''

    beta_lo, beta_hi = get_param_ranges(fit_powerlaw_index=True)
    flag = (beta >= beta_lo) & (beta <= beta_hi)
    if not flag:
        return -np.inf
    else:
        return lnlike_powerlaw(beta, f, fluxmin, fluxmax)

def fleming(f, alpha, f50):
    '''
    The original Fleming completeness function (see eq. 1)

    Parameters
    ----------
    f : float
        linear flux
    alpha : float
        describes the speed with which the completeness declines
    f50 : float
        linear flux at which the completeness fraction is 50%

    Returns
    -------
    fc : float
        the completeness fraction (0, 1)
    '''
    numerator = alpha * np.log10( f / f50 )
    denominator = ( 1. + numerator**2. ) ** 0.5
    return 0.5* (1. + numerator / denominator ) 

def expdecay(x, tau):
    '''
    Exponential decay function

    used for the modification to the Fleming curve (see eq. 2)
    '''
    return 1. - np.exp(-x/tau)

def inverse_fleming(f50, alpha, fcmin=0.1):
    '''
    Find the flux at which the completeness fraction = fcmin
    i.e., inverting eq. 1

    Used for the faint-end modification to the Fleming function (see eq. 2)

    Parameters
    ----------
    f50 : float
        linear flux at which the completeness fraction is 50%
    alpha : float
        describes the speed with which the completeness declines
    fcmin : float in range (0, 1)
        completeness fraction
        in practice, this is the completeness fraction below which the 
        modification to the Fleming curve becomes important (see eq. 3)

    Returns
    -------
    f : float
        linear flux at which the completeness fraction = fcmin
    ''' 

    a = (2*fcmin -1)**2.
    b = -1*( abs(a / (1-a))*alpha**-2. )**0.5

    return f50*10**b

def fleming_modified(f, alpha, f50, fcmin=0.1):
    '''
    The modified Fleming completeness function (see eq. 3)

    Parameters
    ----------
    f : float
        linear flux
    alpha : float
        describes the speed with which the completeness declines
    f50 : float
        linear flux at which the completeness fraction is 50%
    fcmin : float in range (0, 1)
        completeness fraction
        in practice, this is the completeness fraction below which the 
        modification to the Fleming curve becomes important (see eq. 3)

    Returns
    -------
    fc : float
        the completeness fraction (0, 1)
    '''
    f_tau = inverse_fleming(f50=f50, alpha=alpha, fcmin=fcmin)
    fc_decay = expdecay(f, f_tau)
    fc = fleming(f, alpha, f50)
    fc_mod = fc**(1. / fc_decay)
    return fc_mod


def pf_nonnorm(f, beta, f50, alpha, fcmin=0.1, fcut=None):
    '''
    The non-normalized observed flux probability distribution (see eq. 4)

    Parameters
    ----------
    f : float
        linear flux
    beta : float
        the power law index of the underlying flux distribution
    f50 : float
        linear flux at which the completeness fraction is 50%
    alpha : float
        describes the speed with which the completeness declines
    fcmin : float in range (0, 1)
        completeness fraction
        in practice, this is the completeness fraction below which the 
        modification to the Fleming curve becomes important (see eq. 3)
    fcut : None or float
        if float, ignore all fluxes below a low cutoff value

    Returns
    -------
    p : float
        the probability of finding an object with flux f
    '''
    if (isinstance(f, float)) | (isinstance(f, int)):
        f=np.array([f])
    power = f ** -beta
    f_tau = inverse_fleming(f50=f50, alpha=alpha, fcmin=fcmin)
    fc_decay = expdecay(f, f_tau)
    fc = fleming(f=f, alpha=alpha, f50=f50)
    pf = power * fc**(1. / fc_decay)
    if fcut:
        pf[f<fcut] = 0.
    return pf


def get_pf_norm(beta, f50, alpha, fmax=40, fcmin=0.1, fcut=None):
    '''
    find the constant s.t. the probability function integrates to unity

    Parameters
    ----------
    beta : float
        the power law index of the underlying flux distribution
    f50 : float
        linear flux at which the completeness fraction is 50%
    alpha : float
        describes the speed with which the completeness declines
    fmax : float
        high-flux cutoff of the integral 
        (doesn't matter too much, I think, as long as it is within reason,
        i.e., beyond bulk of the observed distribution but not too high)
    fcmin : float in range (0, 1)
        completeness fraction
        in practice, this is the completeness fraction below which the 
        modification to the Fleming curve becomes important (see eq. 3)
    fcut : None or float
        if float, ignore all fluxes below a low cutoff value

    Returns
    -------
    N : float
        normalization constant
    '''
    I = quad(pf_nonnorm, 0, fmax,
                        args=(beta, f50, alpha, fcmin, fcut))[0]
    N = 1. / I
    return N

def pf_norm(f, beta, f50, alpha, fcmin=0.1, fcut=None):
    '''
    The normalized observed flux probability distribution (see eq. 4)
    (normalized <---> the probability integrates to 1

    Parameters
    ----------
    f : float
        linear flux
    beta : float
        the power law index of the underlying flux distribution
    f50 : float
        linear flux at which the completeness fraction is 50%
    alpha : float
        describes the speed with which the completeness declines
    fcmin : float in range (0, 1)
        completeness fraction
        in practice, this is the completeness fraction below which the 
        modification to the Fleming curve becomes important (see eq. 3)
    fcut : None or float
        if float, ignore all fluxes below a low cutoff value

    Returns
    -------
    p : float
        the probability of finding an object with flux f
    '''
    pf = pf_nonnorm(f, beta, f50, alpha, fcmin=fcmin, fcut=fcut)
    N  = get_pf_norm(beta, f50, alpha, fcmin=fcmin, fcut=fcut)
    return N*pf

def lnlike(theta, f, fcmin=0.1, fcut=None):
    '''
    log likelihood for the observed flux distribution

    set of data (f)
    three-parameter model (beta, f50, alpha)

    Parameters
    ----------
    theta : list
        [beta, f50, alpha]
    f : 1d array
        linear fluxes
    fcmin : float in range (0, 1)
        completeness fraction
        in practice, this is the completeness fraction below which the 
        modification to the Fleming curve becomes important (see eq. 3)
    fcut : None or float
        if float, ignore all fluxes below a low cutoff value

    Returns
    -------
    lnlike : float
        log likelihood 
    '''
    beta, f50, alpha = theta
    pf = pf_norm(f, beta, f50, alpha, fcmin=fcmin, fcut=fcut)
    return sum( np.log(pf[pf!=0]) )

def lnprior(theta):
    '''
    Uniform prior for model parameters

    Parameters
    ----------
    theta : list
        [beta, f50, alpha]

    Returns
    -------
    0.0 if all parameters are in bounds, -np.inf if any are out of bounds

    '''
    # check that parameters are in bounds
    beta, alpha = np.array(theta)[ [0, -1] ]
    f50 = theta[1:-1]
    betas, f50s, alphas = get_param_ranges()
    flag = True
    flag *= ( (beta  >= betas[0])  & (beta  <= betas[1]) )
    for f50i in f50:
        flag *= ( (f50i   >= f50s[0])   & (f50i   <= f50s[1])  )
    flag *= ( (alpha >= alphas[0]) & (alpha <= alphas[1]) )

    # ensure that the completeness is sufficiently high at high fluxes, 
    # matching visual inspection 
#    flag *= ( fleming( 8, alpha=alpha, f50=f50 ) > 0.9 ) 

    if not flag:
        return -np.inf
    else:
        return 0.0

def lnprob(theta, f, fcmin=0.1, fcut=None):
    '''
    Calculate the log probability

    Parameters
    ----------
    theta : list
        [beta, f50, alpha]
    f : 1d array
        linear fluxes
          --OR--
        if fitting multiple fields, it is a list of 1d arrays
    fcmin : float in range (0, 1)
        completeness fraction
        in practice, this is the completeness fraction below which the 
        modification to the Fleming curve becomes important (see eq. 3)
    fcut : None or float
        if float, ignore all fluxes below a low cutoff value

    Returns
    -------
    log prior + log likelihood
    '''
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    else:
        # Consider the case where multiple fields are being fit
        try:
            if len(f[0]):
                beta, alpha = np.array(theta)[ [0, -1] ]
                f50 = theta[1:-1]
                lnl = 0.
                for fi, f50i in zip( f, f50 ):
                    thetai = [beta, f50i, alpha]
                    lnli = lnlike(thetai, fi, fcmin=fcmin, fcut=fcut)
                    lnl += lnli
                return lp + lnl
        except TypeError:
            lnl = lnlike(theta, f, fcmin=fcmin, fcut=fcut)
            return lp + lnl

def get_init_walker_values(nwalkers=100, fit_powerlaw_index=False, nf50=1):
    ''' 
    Before running emcee, this function generates starting points
    for each walker in the MCMC process.

    Starting points are randomly drawn from the uniform prior

    Parameters
    ----------
    nwalkers : int
        number of MCMC walkers
    fit_powerlaw_index : bool
        if True, fitting only the power law index at the bright end
                 used to constrain the power law index when fitting full model
        else, fitting the full three-parameter model
    nf50 : int
        number of fields that are being fit

    Returns
    -------
    pos : np.array (2 dim)
        Two dimensional array with Nwalker x Ndim values
    '''
    if fit_powerlaw_index:
        rbeta = get_param_ranges(fit_powerlaw_index)
        betas = np.random.rand(nwalkers) * (rbeta[1] - rbeta[0]) + rbeta[0]
        pos = np.zeros((nwalkers, 1))
        pos[:,0] = betas

    else:
        rbeta, rf50, ralpha = get_param_ranges(fit_powerlaw_index)
        pos = np.zeros((nwalkers, 2 + nf50))
        pos[:,0] = np.random.rand(nwalkers) * (rbeta[1]  - rbeta[0])  + rbeta[0]
        for i in range(nf50):
            pos[:,i+1] = np.random.rand(nwalkers) * (rf50[1]   - rf50[0])   + rf50[0]
        pos[:,-1] = np.random.rand(nwalkers) * (ralpha[1] - ralpha[0]) + ralpha[0]

    return pos

def fit_powerlaw_index(f, fluxmin, fluxmax=40, nwalkers=100, nsteps=1000):
    '''
    Fit the bright end of the flux distribution with a power law

    Parameters
    ----------
    f : 1d array
        linear fluxes
    fluxmin : float
        minimum flux to fit the power law
    fluxmax : float
        maximum flux to fit the power law
    nwalkers : int
        number of MCMC walkers
    nsteps : int
        number of MCMC steps

    Returns
    -------
    samples : 2d array
        all MCMC chains (burn-in removed)
        columns {beta, lnprob}
    '''

    pos0 = get_init_walker_values(nwalkers, fit_powerlaw_index=True)
    ndim = pos0.shape[1]
    start = time.time()
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_powerlaw, args=(f,fluxmin,fluxmax) )

    # Do real run
    sampler.run_mcmc(pos0, nsteps, rstate0=np.random.get_state())
    end = time.time()
    elapsed = end - start
    print("Total time taken: %0.2f s" % elapsed)
    print("Time taken per step per walker: %0.2f ms" %
          (elapsed / nsteps * 1000. / nwalkers))
    # Calculate how long the run should last
    tau = np.max(sampler.acor)
    burnin_step = int(tau*3)
    print("Mean acceptance fraction: %0.2f" %
          (np.mean(sampler.acceptance_fraction)))
    print("AutoCorrelation Steps: %i, Number of Burn-in Steps: %i" %
          (np.round(tau), burnin_step))

    flatchain_length = nwalkers * (nsteps - burnin_step)
    samples = np.zeros((flatchain_length, ndim+1))
    for i in range(ndim):
        flatchains = sampler.chain[:, burnin_step:, i]
        samples[:, i] = flatchains.reshape( (flatchain_length,),order='C')
    flatlnprob = sampler.lnprobability[:, burnin_step:]
    samples[:,-1]  = flatlnprob.reshape((flatchain_length,), order='C')

    return samples


def fit_model(f, fcmin=0.1, fcut=None, nwalkers=100, nsteps=1000,
              fill_vary_f50=True, plt_title='', fnames=None):
    '''
    Fit the flux distribution

    Parameters
    ----------
    f : 1d array
        linear fluxes
          --OR--
        if fitting multiple fields, it is a list of 1d arrays
    fcmin : float in range (0, 1)
        completeness fraction
        in practice, this is the completeness fraction below which the 
        modification to the Fleming curve becomes important (see eq. 3)
    fcut : None or float
        if float, ignore all fluxes below a low cutoff value
    nwalkers : int
        number of MCMC walkers
    nsteps : int
        number of MCMC steps
    fill_vary_f50 : bool
        if True, C.I. on observed flux dist'n plot uses best alpha, beta
    plt_title : str
        print a title on the parameter covariance plot
    fnames : None or list of str
        suffix to differentiate between different flux sets, if fitting multiple fields

    Returns
    -------
    samples : 2d array
        all MCMC chains (burn-in removed)
        columns {beta, f50, alpha, lnprob}
    produce plots:
        a) parameter covariance (triangle)
        b) completeness curve
        c) observed flux distribution + fit

    '''
    # Consider the case where multiple fields are being fit
    try:
        if len(f[0]):
            nf50 = len(f)
    except TypeError:
        f = [f]
        nf50 = 1
    # Label the multiple fields, if appropriate:
    if (nf50>1) & (type(fnames) == type(None)):
        fnames = ['%s'%(i+1) for i in range(nf50)]

    pos0 = get_init_walker_values(nwalkers, nf50=nf50)
    ndim = pos0.shape[1]
    start = time.time()
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(f,fcmin,fcut) )

    # Do real run
    sampler.run_mcmc(pos0, nsteps, rstate0=np.random.get_state())
    end = time.time()
    elapsed = end - start
    print("Total time taken: %0.2f s" % elapsed)
    print("Time taken per step per walker: %0.2f ms" %
          (elapsed / nsteps * 1000. / nwalkers))
    # Calculate how long the run should last
    tau = np.max(sampler.acor)
    burnin_step = int(tau*3)
    print("Mean acceptance fraction: %0.2f" %
          (np.mean(sampler.acceptance_fraction)))
    print("AutoCorrelation Steps: %i, Number of Burn-in Steps: %i" %
          (np.round(tau), burnin_step))

    flatchain_length = nwalkers * (nsteps - burnin_step)
    samples = np.zeros((flatchain_length, ndim+1))
    for i in range(ndim):
        flatchains = sampler.chain[:, burnin_step:, i]
        samples[:, i] = flatchains.reshape( (flatchain_length,),order='C')
    flatlnprob = sampler.lnprobability[:, burnin_step:]
    samples[:,-1]  = flatlnprob.reshape((flatchain_length,), order='C')
    # End of the run

    ### ----------------------------------------------------------------------
    ### ------------------------------ PLOTTING ------------------------------
    ### ----------------------------------------------------------------------

    # plot parameter covariance plots
    fsize = 13
    pnames = [r'$\beta$']
    if nf50==1: 
        pnames.append(r'$f_{50}$')
    else: # account for case of multiple fields
        for n in fnames:
            pnames.append(r'$f_{50,\rm{'+n+'}}$')
    pnames.append(r'$\alpha$')
    # make the plot
    fig = corner.corner(samples[:,:-1], quantiles=[0.16, 0.5, 0.84], 
                        show_titles=True,title_kwargs={"fontsize": fsize},
                        label_kwargs={"fontsize": fsize},labels=pnames)
    # report the assumptions and priors
    pranges = get_param_ranges()
    info  = r'$\beta \in'  + '[%s, %s]'%tuple(pranges[0])+'$\n'
    info += r'$f_{50} \in' + '[%s, %s]'%tuple(pranges[1])+'$\n'
    info += r'$\alpha \in' + '[%s, %s]'%tuple(pranges[2])+'$'
    info += '\n'+r'F$_{C, \tau} = $'+'%i'%int(fcmin*100.)+'%'
    if fcut:
        info+='\n'+r'$f_{\rm cut} =$'+'%0.1f'%fcut

    fig.text(0.7, 0.8, info, fontsize=fsize)
    fig.text(0.48, 0.9, plt_title, fontsize=1.2*fsize)

    for ax in fig.axes:
        ax.minorticks_on()

    # plot the completeness curve and flux distribution
    fc_fluxes = np.linspace(0.5, 30,1000)
    fluxes = np.linspace(0.5,30,5000)
    nbins=70
    bin_df = 5. / 9.
    params = np.percentile(samples[:,:-1],[16,50,84],0)
    beta, alpha = params[1][[0,-1]]
    figcfig = plt.figure()
    figc = figcfig.add_subplot(111)
    figc.minorticks_on()
    f50s = params[:,1:-1]
    i = 0
    for fi, ni in zip(f, fnames):
        if nf50==1:
            lbl='_nolegend_'
            title = plt_title
        else:
            lbl=ni
            title = ni
        f50lo, f50, f50hi = f50s[:,i]
        # plot the completeness curve
        fc = fleming_modified(fc_fluxes, alpha=alpha, f50=f50, fcmin=fcmin)
        fc_lo = fleming_modified(fc_fluxes, alpha=alpha, f50=f50lo, fcmin=fcmin)
        fc_hi = fleming_modified(fc_fluxes, alpha=alpha, f50=f50hi, fcmin=fcmin)
        figc.plot(fc_fluxes, fc, label=lbl,color=orig_palette[i],zorder=2)
        if fill_vary_f50:
            figc.fill_between(fc_fluxes, fc_lo, fc_hi, color=orig_palette[i],
                              alpha=0.2,zorder=1)

        # plot the observed flux distribution
        plt.figure()
        plt.minorticks_on()
        nbins = int(round( (max(fi) - min(fi)) / bin_df ))
        norm = ( len(fi) * np.mean(np.diff(np.histogram(fi,nbins)[1])) )
        fit_lo = norm*pf_norm(f=fluxes, beta=beta, f50=f50lo, alpha=alpha, fcmin=fcmin)
        fit    = norm*pf_norm(f=fluxes, beta=beta, f50=f50,   alpha=alpha, fcmin=fcmin)
        fit_hi = norm*pf_norm(f=fluxes, beta=beta, f50=f50hi, alpha=alpha, fcmin=fcmin)
        if fill_vary_f50:
            plt.fill_between(fluxes, fit_lo, fit_hi, color='grey',alpha=0.7,zorder=2)
        plt.plot(fluxes, fit, c='k',ls='--',zorder=3)
        plt.hist(fi, nbins,color=orig_palette[0],alpha=0.8,edgecolor='k',zorder=1)
        if fcut:
            ylims = plt.gca().get_ylim()
            plt.plot([fcut, fcut], ylims, ls='dotted',c='r',zorder=4)
            plt.ylim(ylims)
        plt.xlabel(r'[O III] $\lambda 5007$ flux ($\times 10^{-17}$ ergs cm$^{-2}$ s$^{-1}$)')
        plt.ylabel('Number of objects')
        plt.xlim([0,30])
#        plt.title(title)
        plt.text(18, 0.8*plt.gca().get_ylim()[1], title)
        plt.tight_layout()
        i+=1
    figc.set_xlabel(r'[O III] $\lambda 5007$ flux ($\times 10^{-17}$ ergs cm$^{-2}$ s$^{-1}$)')
    figc.set_ylabel('Completeness fraction')
    figc.set_xlim([min(fc_fluxes), max(fc_fluxes)])
    figc.set_xscale('log')
    xtloc = [1, 5, 10]
    figc.set_xticks( list(np.array(xtloc)))
    figc.set_xticklabels(xtloc)
    figc.set_ylim([0,1])
    figc.set_title(plt_title)
    figc.legend().draggable()
    figcfig.tight_layout()

    return samples



def inspect_powerlaw_index(f,nbins=100,nwalkers=100,nsteps=1000):
    '''
    explore how the flux range over which the power law is fit
    affects the inferred slope of the power law index
    '''
    def pow_nonnorm(f, beta):
        return f ** -beta

    fluxmax = 25
    fluxmin_list = [5,6,7,8,9,10]
    fluxmin_list=[6,7.5,9]
    beta = []

    # flux bounds to normalize the curve
    inorm = [7, 40]

    # plot in linear flux
    figlin = plt.figure().add_subplot(111)
    plt.minorticks_on()
    figlin.hist(f,nbins,alpha=0.3,color='grey',edgecolor='k')
    plt.xlabel('[O III] $\lambda5007$ flux\n($f_{max} = '+str(fluxmax)+'$)')
    plt.ylabel('Number of objects')
    plt.tight_layout()

    for i,fluxmin in enumerate(fluxmin_list):
        samples = fit_powerlaw_index(f, fluxmin, fluxmax, nwalkers=nwalkers, nsteps=nsteps)
        beta.append( np.percentile(samples[:,0], [16, 50, 84]) )

        # plot the result
        fluxes = np.linspace(fluxmin, fluxmax, 200)
#        b50 = beta[-1][1]
        b16, b50, b84 = beta[-1]
        b16, b84 = np.diff(beta[-1])
        # try to normalize via integral
        pnorm = len(f[(f>=inorm[0])&(f<=inorm[1])])
        pnorm *= np.mean(np.diff(np.histogram(f,nbins)[1]))
        I = quad(pow_nonnorm,inorm[0],inorm[1],args=(b50))[0]
        Nf = pow_nonnorm(fluxes,b50)/I
        Nf *= pnorm
        fluxes2 = np.linspace(5, fluxmin, 100)
        Nf2 = pow_nonnorm(fluxes2,b50)/I
        Nf2 *= pnorm

        figlin.plot(fluxes,Nf,lw=1.8,c=orig_palette[i+1],
                 label=r'$\beta = $'+'%0.1f'%b50+r'$_{'+'%0.1f'%b16+'}^{'+'%0.1f'%b84
                       +'}$\n'+r'$f_{min}=$'+'%0.1f'%fluxmin)
        figlin.plot(fluxes2,Nf2,label='_nolegend_',lw=1.8,ls='--',c=orig_palette[i+1])

    figlin.set_xlim([0,25])
    figlin.set_ylim([0, 1.05 * max(np.histogram(f, nbins)[0]) ])
    figlin.legend().draggable()

    # plot power law index vs flux range
    beta = np.array(beta)
    plt.figure()
    plt.minorticks_on()
    plt.errorbar(fluxmin_list, beta[:,1], yerr=[np.diff(beta)[:,0], np.diff(beta)[:,1]],
                 elinewidth=1., ecolor='grey',fmt='s',ms=6)
    plt.xlabel('$f_{min}$ ($f_{max} = '+str(fluxmax)+'$)')
    plt.ylabel(r'$\beta$')
    plt.tight_layout()

    # ---------- vary the upper flux cutoff ---------------------

    fluxmin = 7
    fluxmax_list=[18,25,33,40]
    beta = []

    # plot in linear flux
    figlin2 = plt.figure().add_subplot(111)
    plt.minorticks_on()
    figlin2.hist(f,nbins,alpha=0.3,color='grey',edgecolor='k')
    plt.xlabel('[O III] $\lambda5007$ flux\n($f_{min} = '+str(fluxmin)+'$)')
    plt.ylabel('Number of objects')
    plt.tight_layout()

    for i,fluxmax in enumerate(fluxmax_list):
        samples = fit_powerlaw_index(f, fluxmin, fluxmax, nwalkers=nwalkers, nsteps=nsteps)
        beta.append( np.percentile(samples[:,0], [16, 50, 84]) )

        # plot the result
        fluxes = np.linspace(fluxmin, fluxmax, 200)
#        b50 = beta[-1][1]
        b16, b50, b84 = beta[-1]
        b16, b84 = np.diff(beta[-1])
        # try to normalize via integral
        pnorm = len(f[(f>=inorm[0])&(f<=inorm[1])])
        pnorm *= np.mean(np.diff(np.histogram(f,nbins)[1]))
        I = quad(pow_nonnorm,inorm[0],inorm[1],args=(b50))[0]
        Nf = pow_nonnorm(fluxes,b50)/I
        Nf *= pnorm
        fluxes2 = np.linspace(fluxmax, 40., 100)
        Nf2 = pow_nonnorm(fluxes2,b50)/I
        Nf2 *= pnorm
        fluxes3 = np.linspace(5, fluxmin, 100)
        Nf3 = pow_nonnorm(fluxes3,b50)/I
        Nf3 *= pnorm

        figlin2.plot(fluxes,Nf,lw=1.8,c=orig_palette[i+1],
                 label=r'$\beta = $'+'%0.1f'%b50+r'$_{'+'%0.1f'%b16+'}^{'+'%0.1f'%b84
                       +'}$\n'+r'$f_{max}=$'+'%0.1f'%fluxmax)
        figlin2.plot(fluxes2,Nf2,label='_nolegend_',lw=1.8,ls='--',c=orig_palette[i+1])
        figlin2.plot(fluxes3,Nf3,label='_nolegend_',lw=1.8,ls='--',c=orig_palette[i+1])

    figlin2.set_xlim([0,25])
    figlin2.set_ylim([0, 1.05 * max(np.histogram(f, nbins)[0]) ])
    figlin2.legend().draggable()

    figlog.set_xlim([0,1.45])
    figlog.set_ylim([0, 1.05 * max(np.histogram(np.log10(f), nbins)[0]) ])
    figlog.legend().draggable()

    beta = np.array(beta)

    plt.figure()
    plt.minorticks_on()
    plt.errorbar(fluxmax_list, beta[:,1], yerr=[np.diff(beta)[:,0], np.diff(beta)[:,1]],
                 elinewidth=1., ecolor='grey',fmt='s',ms=6)
    plt.xlabel('$f_{max}$ ($f_{min} = '+str(fluxmin)+'$)')
    plt.ylabel(r'$\beta$')
    plt.tight_layout()

