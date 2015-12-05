
#!/usr/bin/python

# onlinewikipedia.py: Demonstrates the use of online VB for LDA to
# analyze a bunch of random Wikipedia articles.
#
# Copyright (C) 2010  Matthew D. Hoffman
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import cPickle, string, numpy, getopt, sys, random, time, re, pprint
import matplotlib.pyplot as plt
import onlineldavb
import wikirandom
import printtopics
import time

def main( batchnumber = 3.3e4 ):
    """
    Downloads and analyzes a bunch of random Wikipedia articles using
    online VB for LDA.
    """

    # The number of documents to analyze each iteration
    batchsize = 64*8
    # The total number of documents in Wikipedia
    D = 3.3e6
    # The number of topics
    K = 100

    # How many documents to look at
    #if (len(batchnumber) < 2):
     #   documentstoanalyze = int(D/batchsize)
    #else:
    documentstoanalyze = batchnumber

    # Our vocabulary
    vocab = file('./dictnostops.txt').readlines()
    W = len(vocab)
    # record time used for training
    start = time.clock()
    # Initialize the algorithm with alpha=1/K, eta=1/K, tau_0=1024, kappa=0.7
    olda = onlineldavb.OnlineLDA(vocab, K, D, 1./K, 1./K, 1024., 0.7)
    # Run until we've seen D documents. (Feel free to interrupt *much*
    # sooner than this.)
    perplexity_plot=list()
    for iteration in range(1, documentstoanalyze+1):
        # Download some articles
        (docset, articlenames) = \
            wikirandom.get_random_wikipedia_articles(batchsize)
        # Give them to online LDA
        bound = olda.update_lambda(docset)
        # Compute an estimate of held-out perplexity
        #(wordids, wordcts) = onlineldavb.parse_doc_list(docset, olda._vocab)
        perwordbound = bound * len(docset) / (D * sum(map(sum, olda._wordcts)))
        perplexity_plot.append(numpy.exp(-perwordbound))
        print '%d:  rho_t = %f,  held-out perplexity estimate = %f' % \
            (iteration, olda._rhot, numpy.exp(-perwordbound))

        # Save lambda, the parameters to the variational distributions
        # over topics, and gamma, the parameters to the variational
        # distributions over topic weights for the articles analyzed in
        # the last iteration.
        if (iteration % 10 == 0):
            numpy.savetxt('lambda-%d.dat' % iteration, olda._lambda)
            numpy.savetxt('gamma-%d.dat' % iteration, olda._gamma)
    
    #print time taken
    end = time.clock()
    print "time taken for training %f" %end
    #plot perplexity
    plt.plot(range(len(perplexity_plot)), perplexity_plot, 'g')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Perplexity')
    plt.show()
    plt.pause(100)
    # print topics
    printtopics("dictnostops.txt", "lambda-20.dat")
if __name__ == '__main__':
    #printtopics.main("dictnostops.txt", "lambda-10.dat")
    main(20)
    

