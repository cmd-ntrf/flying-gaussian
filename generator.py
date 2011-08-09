#!/usr/bin/env python2.7
"""Random points generator where the points are sampled from
labeled distributions that can move, rotate and scale along time.
Each sample correspond to one unit of time.
"""
import json
import numpy
import random
import argparse

try:
    import pylab
    from matplotlib.colors import ColorConverter
    from matplotlib.patches import Ellipse
except:
    MATPLOTLIB = False
else:
    MATPLOTLIB = True
    color_conv = ColorConverter()

from collections import namedtuple, deque
from operator import attrgetter, itemgetter
from math import sqrt, cos, sin, pi, atan2
from numpy import linalg
from itertools import imap

# Global constant
LAST_N_PTS = 25
COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

class Distribution(object):
    """Object representing a multivariate normal distribution that 
    changes dynamically. The object contains a list of transformations
    that are applied sequentially, and that modify the scale, the position
    and the angle of the distribution.
    """
    def __init__(self):
        self.update_count = 0
        self.cur_trans = None
        self.scale = 1.0
        
    def sample(self, size=1):
        """Sample the actual distribution *size* times."""
        return numpy.random.multivariate_normal(self.centroid, self.scale * self.matrix, size)
        
    def update(self, time):
        """Update the distribution position, scale and rotation given the current
        time.
        """
        if time < self.start_time:
            return
        # If there is no current transformation
        # pop one from the stack
        if self.cur_trans is None:
            if len(self.transforms) > 0:
                self.cur_trans = self.transforms.pop()
                nbr_steps = self.cur_trans.duration / self.cur_trans.steps
            
                self.delta_t = self.cur_trans.translate / nbr_steps
                
                # Compute rotation delta and rotation matrix
                self.delta_r = self.cur_trans.rotate / nbr_steps * pi / 180.0
                self.rotation = numpy.identity(self.ndim)
                idx = 0
                for i in xrange(self.ndim-1):
                    for j in xrange(i+1, self.ndim):
                        matrix = numpy.identity(self.ndim)
                        angle = self.delta_r[idx]
                        sign = 1.0
                        if i+j % 2 == 0:
                            sign = -1.0
                        matrix[i][i] = cos(angle)
                        matrix[j][j] = cos(angle)
                        matrix[i][j] = -sign*sin(angle)
                        matrix[j][i] = sign*sin(angle)
                        self.rotation = numpy.dot(self.rotation, matrix)
                        idx += 1
                self.rotation_inv = linalg.inv(self.rotation)
                sigma = self.scale * self.cur_trans.scale - self.scale
                self.delta_s = sigma / nbr_steps
            else:
                return
        self.update_count += 1
        # Update the distribution with the current appliable transformation
        if self.update_count % self.cur_trans.steps == 0:
            self.centroid += self.delta_t
            self.scale += self.delta_s
            self.matrix = numpy.dot(self.rotation_inv, numpy.dot(self.matrix, self.rotation))
        
        # Duration of the transformation is over
        if self.update_count == self.cur_trans.duration:
            self.cur_trans = None
            self.update_count = 0

Class = namedtuple('Class', ['label', 'weight', 'distributions', 'start_time'])
Transform = namedtuple('Transform', ['duration', 'steps', 'translate', 'scale', 'rotate'])

def readfile(filename):
    """Read a JSON file containing the classes, the distributions and the transformations, 
    and initialize the corresponding object. The function return a list of initialized classes.
    """
    
    try:
        fp = open(filename)
    except IOError:
        print 'Cannot open file : ', filename
        exit()
        
    jclass_list = json.load(fp)
    class_list = []

    weight_sumc = 0.0
    for jclass in jclass_list:
        weight_sumd = 0.0
        distribs = []
        for jdistrib in jclass['distributions']:
            distr = Distribution()
            distr.start_time = jdistrib['start_time']
            distr.weight = jdistrib['weight']
            
            weight_sumd += distr.weight
            
            distr.centroid = jdistrib['centroid']
            distr.ndim = len(distr.centroid)            
            distr.matrix = numpy.array(jdistrib['cov_matrix'])
            distr.transforms = []
            
            for jtrans in reversed(jdistrib.get('transforms', [])):
                duration = jtrans.get('duration', 1)
                steps = jtrans.get('steps', 1)
                translation = numpy.array(jtrans.get('translate', [0.0] * distr.ndim))
                scale = jtrans.get('scale', 1.0)
                rotate = numpy.array(jtrans.get('rotate', [0.0] * (distr.ndim*(distr.ndim-1)/2)))
                distr.transforms.append(Transform(duration,
                                                  steps,
                                                  translation,
                                                  scale,
                                                  rotate))
            distribs.append(distr)
        
        cstart_time = min(imap(attrgetter('start_time'), distribs))
        class_list.append(Class(jclass['class'], jclass['weight'], distribs, cstart_time))
        weight_sumc += class_list[-1].weight
        if weight_sumd != 1.0:
            print 'Warning: weights sum for distribution of class %s is not equal to one, weights will be normalized.' % class_list[-1].label

    if weight_sumc != 1.0:
        print 'Warning: weights sum the set of classes is not equal to one, weights will be normalized.'

    return class_list

def draw_cov_ellipse(centroid, cov_matrix, sigma, ax, nbr_sigma=2.0, color='b'):
    """Example from matplotlib mailing list :
    http://www.mail-archive.com/matplotlib-users@lists.sourceforge.net/msg14153.html
    """
    U, s, Vh = linalg.svd(cov_matrix)
    orient = atan2(U[1,0],U[0,0])*180.0/pi
    width = nbr_sigma*sigma*sqrt(s[0])
    height = nbr_sigma*sigma*sqrt(s[1])

    ellipse = Ellipse(xy=centroid, width=width, height=height, angle=orient, fc=color)
    ellipse.set_alpha(0.1)
    # pylab.arrow(centroid[0], centroid[1], U[0][0], U[0][1], width=0.02)
    # pylab.arrow(centroid[0], centroid[1], U[1][0], U[1][1], width=0.02)
    
    return ax.add_patch(ellipse)    

def plot_class(time, ref_labels, class_list, points, labels, fig, axis):
    """Plot the distributions ellipses and the last sampled points.
    """
    axis.clear()
    
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    
    for class_ in class_list:
        for distrib in class_.distributions:
            if distrib.start_time <= time:
                min_x = min(min_x, distrib.centroid[0] - 4*distrib.scale*sqrt(distrib.matrix[0][0]))
                max_x = max(max_x, distrib.centroid[0] + 4*distrib.scale*sqrt(distrib.matrix[0][0]))
                min_y = min(min_y, distrib.centroid[1] - 4*distrib.scale*sqrt(distrib.matrix[1][1]))
                max_y = max(max_y, distrib.centroid[1] + 4*distrib.scale*sqrt(distrib.matrix[1][1]))
    
    axis.set_xlim(min_x,max_x)
    axis.set_ylim(min_y,max_y)

    # Draw the last sampled points
    x = map(itemgetter(0), points)
    y = map(itemgetter(1), points)
    alph_inc = 1.0 / len(labels)
    colors = [color_conv.to_rgba(COLORS[ref_labels.index(label)], (i+1)*alph_inc) for i, label in enumerate(labels)]
    axis.scatter(x, y, c=colors, edgecolors='none')

    ellipses = []
    labels = []
    # Draw the distribution covariance ellipse
    for i, class_ in enumerate(class_list):
        present = False
        for distrib in class_.distributions:
            if time >= distrib.start_time:
                ref_ell = draw_cov_ellipse(distrib.centroid, distrib.matrix, distrib.scale, axis, 4.0, COLORS[i])
                if not present:
                    ellipses.append(ref_ell)
                    labels.append(class_.label)
                    present = True

    axis.legend(ellipses, labels)
    fig.canvas.draw()

def weightChoice(seq):
    """Randomly choose an element from the sequence *seq* with a 
    bias function of the weight of each element.
    """
    sorted_seq = sorted(seq, key=attrgetter("weight"), reverse=True)
    sum_weights = sum(elem.weight for elem in seq)
    u = random.random() * sum_weights
    sum_ = 0.0
    for elem in sorted_seq:
        sum_ += elem.weight
        if sum_ >= u:
            return elem

def main(filenae, samples, plot):
    if plot and not MATPLOTLIB:
        print 'Warning: the \'plot\' option was activated, but matplotlib is unavailable. Processing will continue without plotting.'    
    
    # Read file and initialize classes
    class_list = readfile(args.filename)
    
    # Initialize figure and axis before plotting
    if plot and MATPLOTLIB:
        fig = pylab.figure(figsize=(10,10))
        ax1 = pylab.subplot2grid((3,3), (0,0), colspan=3, rowspan=3)
        pylab.ion()
        pylab.show()
        points = deque(maxlen=LAST_N_PTS)
        labels = deque(maxlen=LAST_N_PTS)
        ref_labels = map(attrgetter('label'), class_list)

    for i in xrange(samples):
        class_ = weightChoice([class_ for class_ in class_list if i >= class_.start_time])
        distrib = weightChoice([distrib for distrib in class_.distributions if i >= distrib.start_time])
        spoint =  distrib.sample()[0]
        
        # Print the sampled point in CSV format
        print "%s, %s" % (str(class_.label), ", ".join(map(str, spoint)))
        
        # Plot the resulting distribution if required
        if plot and MATPLOTLIB:
            points.append(spoint)
            labels.append(class_.label)
            plot_class(i, ref_labels, class_list, points, labels, fig, ax1)
            # fig.savefig('images2/point_%i.png' % i)
        
        # Update the classes' distributions
        for class_ in class_list:
            for distrib in class_.distributions:
                distrib.update(i)

    if plot and MATPLOTLIB:
          pylab.ioff()
          pylab.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read a file of classes and return a series of randomly sampled points from those classes.')
    parser.add_argument('filename', help='an integer for the accumulator')
    parser.add_argument('samples', type=int, help='number of samples')
    
    parser.add_argument('--plot', dest='plot', required=False, action='store_true',
                        help='tell if the results should be plotted')
    args = parser.parse_args()
    main(args.filename, args.samples, args.plot)
