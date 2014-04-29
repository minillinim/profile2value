#!/usr/bin/env python
###############################################################################
#
#    colDiff.py
#
#    Work out the similarity between columns in large matrices of data
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

import pylab
import copy
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

import matplotlib.pyplot as plt
from matplotlib import colorbar
from mpl_toolkits.mplot3d import Axes3D, axes3d, proj3d

from colorsys import hsv_to_rgb as htr
import gzip
import mimetypes

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

    CD_open = open
    try:
        # handle gzipped files
        mime = mimetypes.guess_type(args.infile)
        if mime[1] == 'gzip':
            CD_open = gzip.open
    except:
        print "Error when guessing contig file mimetype"
        raise

    headers = None
    with CD_open(args.infile, 'r') as inCSV:
        if(args.header):
            line = next(inCSV)
            row = line.rstrip().split(args.sep)
            headers = row[1:]

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
    print "Keeping %d rows across %d cols with row sum at least %d" % (num_rows, num_cols, args.rowsum+1)

    dists = []
    if (args.subset != 0) and (args.subset < num_rows):
        # make subsets of the data
        num_subsets = int(float(num_rows) / float(args.subset))
        indices = np.arange(num_rows)
        np.random.shuffle(indices)

        for i in np.arange(num_subsets):
            sub_data = np.transpose(np.copy(data_array[indices[i*args.subset:(i+1)*args.subset],:]))
            Center(sub_data,verbose=0)
            p = PCA(sub_data)
            components = p.pc()[:,0:3]

            # scale PCA
            min_score = np.min(components, axis=0)
            components -= min_score
            max_score = np.max(components, axis=0)
            components /= max_score

            try:
                dists += squareform(pdist(components))
            except ValueError:
                dists = squareform(pdist(components))

            """
            fig = plt.figure()
            ax = plt.subplot(1,1,1, projection='3d')
            ax.scatter(components[:,0],
                       components[:,1],
                       components[:,2],
                       s=0.1)
            plt.show()
            plt.close(fig)
            del fig
            """
        dists /= float(num_subsets)

    else:
        sub_data = np.transpose(np.copy(data_array))
        Center(sub_data,verbose=0)
        p = PCA(sub_data)
        components = p.pc()[:,0:3]

        # scale PCA
        min_score = np.min(components, axis=0)
        components -= min_score
        max_score = np.max(components, axis=0)
        components /= max_score

        dists = squareform(pdist(components))


    # plot
    fig = plt.figure()
    axisHeatMap = fig.add_axes([0.1, 0.15, 0.8, 0.85], frame_on=False)
    axisColourMap = fig.add_axes([0.2, 0.05, 0.6, 0.05], frame_on=False)
    colormap = pylab.cm.PiYG

    # heat map
    X = []
    Y = []
    for i in np.arange(num_cols):
        for j in np.arange(num_cols):
            X.append(i)
            Y.append(j)
    sc = axisHeatMap.scatter(X,
                             Y[::-1],
                             edgecolors='none',
                             c=dists.ravel(),
                             cmap=colormap,
                             s=1000,
                             marker='s')
    axisHeatMap.set_xticklabels([])
    axisHeatMap.set_yticklabels([])
    axisHeatMap.set_xticks([])
    axisHeatMap.set_yticks([])

    # colour bar
    min_value = np.min(dists)
    max_value = np.max(dists)

    colour_bar = colorbar.ColorbarBase(axisColourMap, cmap=colormap, orientation='horizontal')
    colour_bar.set_ticks([0, 0.5, 1])
    colour_bar.set_ticklabels(['%.4f' % (min_value), '%.4f' % (0.5*(max_value+min_value)), '%.4f' % (max_value)])


    # row and column labels
    for i in np.arange(num_cols):
        axisHeatMap.text(num_cols - 0.5, i, '  ' + headers[::-1][i], horizontalalignment="left")

    for i in np.arange(num_cols):
        axisHeatMap.text(i, -0.5, '  ' +headers[i], rotation = 270, verticalalignment="top")

    if args.outfile != "":
        try:
            fig.set_size_inches(12,12)
            plt.savefig(args.outfile,dpi=300)
            plt.close(fig)
        except:
            print "Error saving image",args.outfile, sys.exc_info()[0]
            raise
    else:
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
    parser.add_argument('-o', '--outfile', default="", help="name of image file to make. None for show plot")
    parser.add_argument('-m', '--maxrows', type=int, default=0, help="load the first m rows (0 for all rows)")
    parser.add_argument('-S', '--subset', type=int, default=10000, help="do several PCAs random subsets of rows (for _LARGE_ datasets - 0 for all rows)")
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


