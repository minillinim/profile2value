#!/usr/bin/env python
###############################################################################
#
#    profile2view.py
#
#    Convert a descriptive profile into a PCA plot
#
#    Copyright (C) 2014 Michael Imelfort
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
###############################################################################

import argparse
import sys
from pprint import pprint

import copy
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, axes3d, proj3d

from colorsys import hsv_to_rgb as htr

###############################################################################
# CODE HERE
###############################################################################

def doWork( args ):
    # open the input and output files
    data_array = []
    num_rows = 0
    num_cols = 0

    # convert the input csv into an array
    if args.verbose:
        print "loading data..."
    with open(args.infile, 'r') as inCSV:
        if(args.header):
            next(inCSV)

        # get the first "real" line to work out the number of columns
        for line in inCSV:
            row = line.rstrip().split(args.sep)
            data_array.append([float(x) for x in row[1:]])
            if(0 == num_rows):
                num_cols = len(row)
                break

        if args.verbose:
            num_rows = 1
            count_rows = 1
            print_at = 50000
            for line in inCSV:
                if args.maxrows != 0 and num_rows >= args.maxrows:
                    break
                row = line.rstrip().split(args.sep)
                data_array.append([float(x) for x in row[1:]])
                num_rows += 1
                count_rows += 1
                if count_rows == print_at:
                    count_rows = 0
                    print "%d rows loaded" % num_rows

        else:
            num_rows = 1
            for line in inCSV:
                if args.maxrows != 0 and num_rows >= args.maxrows:
                    break
                row = line.rstrip().split(args.sep)
                data_array.append([float(x) for x in row[1:]])
                num_rows += 1

    # adjust for the row names
    num_cols -= 1
    data_array = np.array(data_array).reshape((num_rows, num_cols))
    print "Loaded" ,num_rows, "non-zero rows across" , num_cols, "cols"

    if args.cols:
        data_array = data_array[:,args.cols]

    bad_rows = np.nonzero(data_array.sum(axis=1) <= args.rowsum)
    data_array = np.delete(data_array, bad_rows, axis=0)
    (num_rows, num_cols) = np.shape(data_array)
    print "Keeping %d rows across %d cols wit row sum at least %d" % (num_rows, num_cols, args.rowsum+1)

    data_array = np.sqrt(data_array)

    # do the PCA and extract the scores
    if args.remove != 0.:
        dc = np.copy(data_array)
    if (args.subset != 0) and (args.subset < num_rows):
        indices = np.arange(num_rows)
        np.random.shuffle(indices)
        nd = np.copy(data_array[indices[:args.subset]])
        if args.verbose:
            print "Performing PCA on %d rows" % len(nd)
        Center(nd,verbose=0)
        p = PCA(nd)
        if args.verbose:
            print "Extending PCA to all non-zero rows"
        Center(data_array,verbose=0)
        components = p.extend(data_array)[:,0:3]
    else:
        # do on the whole thing
        if args.verbose:
            print "Performing PCA on %d rows" % num_rows
        Center(data_array,verbose=0)
        p = PCA(data_array)
        components = p.pc()[:,0:3]

    # remove x% outliers
    if args.remove != 0.:
        data_array = dc
        if args.verbose:
            print "Removing %0.4f percent of the outlying data" % (args.remove*100)
        centroid = np.mean(components, axis=0)
        dists = np.ravel(cdist(components, [centroid]))
        num_to_cull = num_rows * args.remove
        s_d = np.argsort(dists)[::-1]
        data_array = data_array[s_d[num_to_cull:]]
        (num_rows, num_cols) = np.shape(data_array)
        print "Remaining" ,num_rows, "non-zero rows across" , num_cols, "cols"

        # do the PCA (again) and extract the scores
        if (args.subset != 0) and (args.subset < num_rows):
            indices = np.arange(num_rows)
            np.random.shuffle(indices)
            nd = np.copy(data_array[indices[:args.subset]])
            if args.verbose:
                print "(re)Performing PCA on %d rows" % len(nd)
            Center(nd,verbose=0)
            p = PCA(nd)
            if args.verbose:
                print "(re)Extending PCA to all non-zero rows"
            Center(data_array,verbose=0)
            components = p.extend(data_array)[:,0:3]
        else:
            # do on the whole thing
            if args.verbose:
                print "(re)Performing PCA on %d rows" % num_rows
            Center(data_array,verbose=0)
            p = PCA(data_array)
            components = p.pc()[:,0:3]


    # scale PCA
    min_score = np.min(components, axis=0)
    components -= min_score
    max_score = np.max(components, axis=0)
    components /= max_score

    fig = plt.figure()
    ax = plt.subplot(1,1,1, projection='3d')
    ax.scatter(components[:,0],
               components[:,1],
               components[:,2],
               s=0.1)
    plt.show()
    plt.close(fig)
    del fig

class PCA:
    """http://stackoverflow.com/questions/1730600/principal-component-analysis-in-python"""
    def __init__( self, A, fraction=0.90 ):
        assert 0 <= fraction <= 1
            # A = U . diag(d) . Vt, O( m n^2 ), lapack_lite --
        self.U, self.d, self.Vt = np.linalg.svd( A, full_matrices=False )
        assert np.all( self.d[:-1] >= self.d[1:] )  # sorted
        self.eigen = self.d**2
        self.sumvariance = np.cumsum(self.eigen)
        self.sumvariance /= self.sumvariance[-1]
        self.npc = np.searchsorted( self.sumvariance, fraction ) + 1
        while(self.npc < 3):   # prevents less than 3 pcs being found
            fraction *= 1.1
            self.npc = np.searchsorted( self.sumvariance, fraction ) + 1
        self.dinv = np.array([ 1/d if d > self.d[0] * 1e-6  else 0
                                for d in self.d ])
        self.magicMatrix = np.linalg.inv(self.Vt)

    def extend(self, B):
        """Extend the results calculated from PCA'ing a subset to an
        antire matrix. Useful for PCA'ing LARGE datasets"""
        return np.dot(B, self.magicMatrix)

    def pc( self ):
        """ e.g. 1000 x 2 U[:, :npc] * d[:npc], to plot etc. """
        n = self.npc
        return self.U[:, :n] * self.d[:n]

    # These 1-line methods may not be worth the bother;
    # then use U d Vt directly --

    def vars_pc( self, x ):
        n = self.npc
        return self.d[:n] * np.dot( self.Vt[:n], x.T ).T  # 20 vars -> 2 principal

    def pc_vars( self, p ):
        n = self.npc
        return np.dot( self.Vt[:n].T, (self.dinv[:n] * p).T ) .T  # 2 PC -> 20 vars

    def pc_obs( self, p ):
        n = self.npc
        return np.dot( self.U[:, :n], p.T )  # 2 principal -> 1000 obs

    def obs_pc( self, obs ):
        n = self.npc
        return np.dot( self.U[:, :n].T, obs ) .T  # 1000 obs -> 2 principal

    def obs( self, x ):
        return self.pc_obs( self.vars_pc(x) )  # 20 vars -> 2 principal -> 1000 obs

    def vars( self, obs ):
        return self.pc_vars( self.obs_pc(obs) )  # 1000 obs -> 2 principal -> 20 vars

class Center:
    """http://stackoverflow.com/questions/1730600/principal-component-analysis-in-python"""
    """ A -= A.mean() /= A.std(), inplace -- use A.copy() if need be
        uncenter(x) == original A . x
    """
        # mttiw
    def __init__( self, A, axis=0, scale=True, verbose=1 ):
        self.mean = A.mean(axis=axis)
        if verbose:
            print "Center -= A.mean:", self.mean
        A -= self.mean
        if scale:
            std = A.std(axis=axis)
            self.std = np.where( std, std, 1. )
            if verbose:
                print "Center /= A.std:", self.std
            A /= self.std
        else:
            self.std = np.ones( A.shape[-1] )
        self.A = A

    def uncenter( self, x ):
        return np.dot( self.A, x * self.std ) + np.dot( x, self.mean )

###############################################################################
# TEMPLATE SUBS
###############################################################################
#
# Entry point, parse command line args and call out to doWork
#
if __name__ == '__main__':

    # intialise the args parser
    parser = argparse.ArgumentParser(description='Convert a descriptive profile into a single value between 0 -> 1',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # add args here:
    parser.add_argument('infile', help="CSV file of profiles to parse")
    parser.add_argument('-m', '--maxrows', type=int, default=0, help="load the first m rows (0 for all rows)")
    parser.add_argument('-S', '--subset', type=int, default=100000, help="do pca on random subset of rows (for _LARGE_ datasets - 0 for all rows)")
    parser.add_argument('-r', '--remove', type=float, default=0.05, help="Remove the furthest r percent outliers from the dataset")
    parser.add_argument('-R', '--rowsum', type=int, default=0, help="remove all rows with a sum less than or equal to this")
    parser.add_argument('-s', '--sep', default="\t", help="Separator")
    parser.add_argument('-H', '--header', action="store_true", default=False, help="Does the file have a header")
    parser.add_argument('-v', '--verbose', action="store_true", default=False, help="Be V-E-R-B-O-S-E")
    parser.add_argument('-c', '--cols', nargs='+', default=None, help="Only use these colums None for all colums")

    # get and check args
    args = parser.parse_args()

    # do what we came here to do
    #
    doWork(args)


