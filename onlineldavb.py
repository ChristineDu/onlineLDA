# onlineldavb.py: Package of functions for fitting Latent Dirichlet
# Allocation (LDA) with online variational Bayes (VB) in parallel.
#


import sys, re, time, string, threading
import numpy as n
from scipy.special import gammaln, psi

n.random.seed(100000001)
meanchangethresh = 0.001

def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return(psi(alpha) - psi(n.sum(alpha)))
    return(psi(alpha) - psi(n.sum(alpha, 1))[:, n.newaxis])

def parse_doc_list(docs, vocab):
    """
    Parse a document into a list of word ids and a list of counts,
    or parse a set of documents into two lists of lists of word ids
    and counts.

    Arguments: 
    docs:  List of D documents. Each document must be represented as
           a single string. (Word order is unimportant.) Any
           words not in the vocabulary will be ignored.
    vocab: Dictionary mapping from words to integer ids.

    Returns a pair of lists of lists. 

    The first, self._wordids, says what vocabulary tokens are present in
    each document. self._wordids[i][j] gives the jth unique token present in
    document i. (Don't count on these tokens being in any particular
    order.)

    The second, self._wordcts, says how many times each vocabulary token is
    present. self._wordcts[i][j] is the number of times that the token given
    by self._wordids[i][j] appears in document i.
    """
    if (type(docs).__name__ == 'str'):
        temp = list()
        temp.append(docs)
        docs = temp

    D = len(docs)
    print D
    wordids = list()
    wordcts = list()

    it = 0       
    maxthreads = 8
    threads = []
    parselock = threading.Lock()
    
    def parsethread(parselock, it):
        for d in range(it*64, it*64+64):
            docs[d] = docs[d].lower()
            docs[d] = re.sub(r'-', ' ', docs[d])
            docs[d] = re.sub(r'[^a-z ]', '', docs[d])
            docs[d] = re.sub(r' +', ' ', docs[d])
            words = string.split(docs[d])
            ddict = dict()
            for word in words:
                if (word in vocab):
                    wordtoken = vocab[word]
                    if (not wordtoken in ddict):
                        ddict[wordtoken] = 0
                    ddict[wordtoken] += 1
            parselock.acquire()
            wordids.append(ddict.keys())
            wordcts.append(ddict.values())
            parselock.release()

    for it in range(0, maxthreads):
        t = threading.Thread(target = parsethread, args = (parselock, it))
        threads.append(t)
        t.start()
    for it in range(0, maxthreads):
        threads[it].join()        
    return((wordids, wordcts))

class OnlineLDA:
    """
    Implements online VB for LDA as described in (Hoffman et al. 2010).
    """

    def __init__(self, vocab, K, D, alpha, eta, tau0, kappa):
        """
        Arguments:
        K: Number of topics
        vocab: A set of words to recognize. When analyzing documents, any word
           not in this set will be ignored.
        D: Total number of documents in the population. For a fixed corpus,
           this is the size of the corpus. In the truly online setting, this
           can be an estimate of the maximum number of documents that
           could ever be seen.
        alpha: Hyperparameter for prior on weight vectors theta
        eta: Hyperparameter for prior on topics beta
        tau0: A (positive) learning parameter that downweights early iterations
        kappa: Learning rate: exponential decay rate---should be between
             (0.5, 1.0] to guarantee asymptotic convergence.

        Note that if you pass the same set of D documents in every time and
        set kappa=0 this class can also be used to do batch VB.
        """
        self._vocab = dict()
        for word in vocab:
            word = word.lower()
            word = re.sub(r'[^a-z]', '', word)
            self._vocab[word] = len(self._vocab)

        self._K = K
        self._W = len(self._vocab)
        self._D = D
        self._alpha = alpha
        self._eta = eta
        self._tau0 = tau0 + 1
        self._kappa = kappa
        self._updatect = 0

        # Initialize the variational distribution q(beta|lambda)
        self._lambda = 1*n.random.gamma(100., 1./100., (self._K, self._W))
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)

        
    
    


    def do_e_step(self, docs):
        """
        Given a mini-batch of documents, estimates the parameters
        gamma controlling the variational distribution over the topic
        weights for each document in the mini-batch.

        Arguments:
        docs:  List of D documents. Each document must be represented
               as a string. (Word order is unimportant.) Any
               words not in the vocabulary will be ignored.

        Returns a tuple containing the estimated values of gamma,
        as well as sufficient statistics needed to update lambda.
        """
        # Initialize the variantions in do_e_step, batchD is 64
        self._sstats = n.zeros(self._lambda.shape)
        self._gamma = 1*n.random.gamma(100., 1./100., (64*8, self._K))
        # This is to handle the case where someone just hands us a single
        # document, not in a list.
        if (type(docs).__name__ == 'string'):
            temp = list()
            temp.append(docs)
            docs = temp

        (self._wordids, self._wordcts) = parse_doc_list(docs, self._vocab)

        # Now, for each document d update that document's gamma and phi
        it = 0       
        maxthreads = 8
        threads = []
        lock = threading.Lock()

        Elogtheta = dirichlet_expectation(self._gamma)
        expElogtheta = n.exp(Elogtheta)
        def e_step_threading(lock, it):
        
        # Initialize the variational distribution q(theta|gamma) for
        # the mini-batch
        #gamma = 1*n.random.gamma(100., 1./100., (batchD, self._K))
            
            for d in range(it*64, it*64 + 64):
                # These are mostly just shorthand (but might help cache locality)
                ids = self._wordids[d]
                cts = self._wordcts[d]
                gammad = self._gamma[d, :]
                Elogthetad = Elogtheta[d, :]
                expElogthetad = expElogtheta[d, :]
                expElogbetad = self._expElogbeta[:, ids]
                # The optimal phi_{dwk} is proportional to 
                # expElogthetad_k * expElogbetad_w. phinorm is the normalizer.
                phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100
                # Iterate between gamma and phi until convergence
                for it in range(0, 100):
                    lastgamma = gammad
                    # We represent phi implicitly to save memory and time.
                    # Substituting the value of the optimal phi back into
                    # the update for gamma gives this update. Cf. Lee&Seung 2001.
                    gammad = self._alpha + expElogthetad * \
                        n.dot(cts / phinorm, expElogbetad.T)
                    Elogthetad = dirichlet_expectation(gammad)
                    expElogthetad = n.exp(Elogthetad)
                    phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100
                    # If gamma hasn't changed much, we're done.
                    meanchange = n.mean(abs(gammad - lastgamma))
                    if (meanchange < meanchangethresh):
                        break
                lock.acquire()
                self._gamma[d, :] = gammad
                # Contribution of document d to the expected sufficient
                # statistics for the M step.
                self._sstats[:, ids] += n.outer(expElogthetad.T, cts/phinorm)
                lock.release()
        
        for it in range(0, maxthreads):
            t = threading.Thread(target = e_step_threading, args = (lock, it))
            threads.append(t)
            t.start()
        for it in range(0, maxthreads):
            threads[it].join()
        
        # This step finishes computing the sufficient statistics for the
        # M step, so that
        # sstats[k, w] = \sum_d n_{dw} * phi_{dwk} 
        # = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.
        self._sstats = self._sstats * self._expElogbeta

    
    def update_lambda(self, docs):
        """
        First does an E step on the mini-batch given in self._wordids and
        self._wordcts, then uses the result of that E step to update the
        variational parameter matrix lambda.

        Arguments:
        docs:  List of D documents. Each document must be represented
               as a string. (Word order is unimportant.) Any
               words not in the vocabulary will be ignored.

        Returns gamma, the parameters to the variational distribution
        over the topic weights theta for the documents analyzed in this
        update.

        Also returns an estimate of the variational bound for the
        entire corpus for the OLD setting of lambda based on the
        documents passed in. This can be used as a (possibly very
        noisy) estimate of held-out likelihood.
        """

        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.
        rhot = pow(self._tau0 + self._updatect, -self._kappa)
        self._rhot = rhot
        # Do an E step to update gamma, phi | lambda for this
        # mini-batch. This also returns the information about phi that
        # we need to update lambda.
        
        self.do_e_step(docs)
        
        # Estimate held-out likelihood for current values of lambda.
        bound = self.approx_bound(docs)
        # Update lambda based on documents.
        self._lambda = self._lambda * (1-rhot) + \
            rhot * (self._eta + self._D * self._sstats / len(docs))
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)
        self._updatect += 1

        return bound

    def approx_bound(self, docs):
        """
        Estimates the variational bound over *all documents* using only
        the documents passed in as "docs." gamma is the set of parameters
        to the variational distribution q(theta) corresponding to the
        set of documents passed in.

        The output of this function is going to be noisy, but can be
        useful for assessing convergence.
        """

        # This is to handle the case where someone just hands us a single
        # document, not in a list.
        if (type(docs).__name__ == 'string'):
            temp = list()
            temp.append(docs)
            docs = temp

        #(self._wordids, self._wordcts) = parse_doc_list(docs, self._vocab)
        batchD = len(docs)

        score = 0
        Elogtheta = dirichlet_expectation(self._gamma)
        expElogtheta = n.exp(Elogtheta)

        # E[log p(docs | theta, beta)]
        for d in range(0, batchD):
            gammad = self._gamma[d, :]
            ids = self._wordids[d]
            cts = n.array(self._wordcts[d])
            phinorm = n.zeros(len(ids))
            for i in range(0, len(ids)):
                temp = Elogtheta[d, :] + self._Elogbeta[:, ids[i]]
                tmax = max(temp)
                phinorm[i] = n.log(sum(n.exp(temp - tmax))) + tmax
            score += n.sum(cts * phinorm)
        score += n.sum((self._alpha - self._gamma)*Elogtheta)
        score += n.sum(gammaln(self._gamma) - gammaln(self._alpha))
        score += sum(gammaln(self._alpha*self._K) - gammaln(n.sum(self._gamma, 1)))

        # Compensate for the subsampling of the population of documents
        score = score * self._D / len(docs)

        # E[log p(beta | eta) - log q (beta | lambda)]
        score = score + n.sum((self._eta-self._lambda)*self._Elogbeta)
        score = score + n.sum(gammaln(self._lambda) - gammaln(self._eta))
        score = score + n.sum(gammaln(self._eta*self._W) - 
                              gammaln(n.sum(self._lambda, 1)))

        return(score)

