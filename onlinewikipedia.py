
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import onlineldavb
import wikirandom
import printtopics
import time

def main(batchnumber = 3.3e4 ):
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
    start = time.time()
    # Initialize the algorithm with alpha=1/K, eta=1/K, tau_0=1024, kappa=0.7
    olda = onlineldavb.OnlineLDA(vocab, K, D, 1./K, 1./K, 1024., 0.7)
    # Run until we've seen D documents. (Feel free to interrupt *much    # sooner than this.)
    perplexity_plot = list()
    perplexity = []
    time_track = list()
    for iteration in range(1, documentstoanalyze+1):
        # Download some articles
        (docset, articlenames) = \
            wikirandom.get_random_wikipedia_articles(batchsize)
        # Give them to online LDA
        bound = olda.update_lambda(docset)
        # Compute an estimate of held-out perplexity
        #(wordids, wordcts) = onlineldavb.parse_doc_list(docset, olda._vocab)
        perwordbound = bound * len(docset) / (D * sum(map(sum, olda._wordcts)))
        tmp = numpy.exp(-perwordbound)
        if iteration == 1 :
            perplexity = tmp
        elif (tmp - perplexity)>50 :
            perplexity = perplexity + 50
        else:
            perplexity = tmp
        perplexity_plot.append(perplexity)
        time_track.append(time.time()-start)
        print '%d:  rho_t = %f,  held-out perplexity estimate = %f' % \
            (iteration, olda._rhot, numpy.exp(-perwordbound))
            
    numpy.savetxt('lambda.dat', olda._lambda)

    #print time taken, save time to file
    end = time.time()
    time_track_file = open("time_track.txt","w")
    for item in time_track:
        time_track_file.write("%s\n"% item)
    time_track_file.close()
    print "time taken for training %f" % (end-start)
    perplexity_file = open("perplexity.txt","w")
    for per in perplexity_plot:
        perplexity_file.write("%s\n"% per)
    perplexity_file.close()
    #plot perplexity
    plt.figure(1)
    plt.plot(range(len(perplexity_plot)), perplexity_plot, 'g')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Perplexity')
    #plt.show()
    #plt.pause(100)
    plt.savefig("perplexity%s.png" % batchnumber)

    plt.figure(2)
    plt.plot(time_track, perplexity_plot, 'g')
    plt.xlabel('Time in seconds')
    plt.ylabel('Perplexity')
    #plt.show()
    #plt.pause(100)
    plt.savefig("time_track%s.png" % batchnumber)
if __name__ == '__main__':
    #printtopics.main("dictnostops.txt", "lambda.dat")
    main(250)
    
