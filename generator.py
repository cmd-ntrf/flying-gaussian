#!/usr/bin/env python2.7
"""Random points generator where the points are sampled from
labeled distributions that can move, rotate and scale along time.
Each sample correspond to one unit of time.
"""

import argparse
import json
import random


import numpy

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ColorConverter
    from matplotlib.patches import Ellipse
except ImportError:
    MATPLOTLIB = False
else:
    MATPLOTLIB = True
    color_conv = ColorConverter()

from collections import namedtuple, deque
from itertools import imap, repeat
from math import sqrt, cos, sin, pi, atan2, exp
from operator import attrgetter, itemgetter, div

from numpy import linalg
from numpy.random import multivariate_normal
from scipy.stats import chi2

# Global constant
LAST_N_PTS = 25
COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

class Class(object):
    """Object representing a set of distribution associated to the same
    label.
    """
    def __init__(self, label, weight, distribs, start_time, 
                 transforms=None):
        self.label = label
        self.weight = weight
        self.distributions = distribs
        self.start_time = start_time
        self.update_count = 0
        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = []
        self.cur_trans = None

    def update(self, time):
        """Upates class' distribution then apply current transform.
        """
        # Update distributions
        for distrib in self.distributions:
            distrib.update(time)
        
        if self.cur_trans is None:
            if len(self.transforms) > 0:
                # Compute the nbr of steps
                self.cur_trans = self.transforms.pop()
                nbr_steps = self.cur_trans.duration / self.cur_trans.steps
                
                # Compute the weight delta
                self.delta_w = (self.cur_trans.weight - self.weight) / float(nbr_steps)
            else:
                return
        
        self.update_count += 1
        # Update the distribution with the current appliable transformation
        if self.update_count % self.cur_trans.steps == 0:
            self.weight += self.delta_w
        
        # Duration of the transformation is over
        if self.update_count == self.cur_trans.duration:
            self.cur_trans = None
            self.update_count = 0

class Distribution(object):
    """Object representing a multivariate normal distribution that 
    changes dynamically. The object contains a list of transformations
    that are applied sequentially, and that modify the scale, the position
    and the angle of the distribution.
    """
    def __init__(self):
        self.update_count = 0
        self.transforms = []
        self.cur_trans = None
        self.scale = 1.0

    def pdf(self, point):
        """Compute the probability to sample a *point*."""
        prob = 1.0/sqrt(2*pi*linalg.det(self.scale*self.matrix))
        prob *= exp(-0.5 * numpy.dot((point - self.centroid).T,
                                     numpy.dot(linalg.inv(self.scale*self.matrix),
                                               point - self.centroid)))
        return prob

       
    def sample(self, size=1):
        """Sample the actual distribution *size* times."""
        return multivariate_normal(self.centroid, self.scale*self.matrix, 
                                   size)
        
    def update(self, time):
        """Update the distribution position, scale and rotation given the 
        current time.
        """
        if time < self.start_time:
            return
        # If there is no current transformation
        # pop one from the stack
        if self.cur_trans is None:
            if len(self.transforms) > 0:
                # Compute the translation delta
                self.cur_trans = self.transforms.pop()
                nbr_steps = self.cur_trans.duration / self.cur_trans.steps
                self.delta_t = self.cur_trans.translate / nbr_steps
                
                # Compute rotation delta and rotation matrix
                self.delta_r = self.cur_trans.rotate / nbr_steps * pi/180.0
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

                # Compute the scaling delta
                sigma = self.scale * self.cur_trans.scale - self.scale
                self.delta_s = sigma / nbr_steps

                # Compute the weight delta
                self.delta_w = (self.cur_trans.weight - self.weight) / nbr_steps
            else:
                return

        self.update_count += 1
        # Update the distribution with the current appliable transformation
        if self.update_count % self.cur_trans.steps == 0:
            self.centroid += self.delta_t
            self.scale += self.delta_s
            self.matrix = numpy.dot(self.rotation_inv, 
                                    numpy.dot(self.matrix, self.rotation))
            self.weight += self.delta_w

        # Duration of the transformation is over
        if self.update_count == self.cur_trans.duration:
            self.cur_trans = None
            self.update_count = 0

Transform = namedtuple('Transform', ['duration', 'steps', 'translate', 
                                     'scale', 'rotate', 'weight'])

def read_file(filename):
    """Read a JSON file containing the classes, the distributions and the
    transformations, and initialize the corresponding object. The function 
    return a list of initialized classes.
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
        transforms = []
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
                translate = jtrans.get('translate', [0.0] * distr.ndim)
                translation = numpy.array(translate)
                scale = jtrans.get('scale', 1.0)
                nbr_angles = distr.ndim*(distr.ndim-1)/2
                rotate = jtrans.get('rotate', [0.0] * nbr_angles)
                rotation = numpy.array(rotate)
                weight = jtrans.get('weight', distr.weight)
                distr.transforms.append(Transform(duration,
                                                  steps,
                                                  translation,
                                                  scale,
                                                  rotation,
                                                  weight))
            distribs.append(distr)

        cstart_time = min(imap(attrgetter('start_time'), distribs))
        label = jclass['class']
        cl_weight = jclass['weight']
        for jtrans in reversed(jclass.get('transforms', [])):
            duration = jtrans.get('duration', 1)
            steps = jtrans.get('steps', 1)
            weight = jtrans.get('weight', cl_weight)
            transforms.append(Transform(duration,
                                        steps,
                                        None,
                                        None,
                                        None,
                                        weight))

        class_list.append(Class(label, cl_weight, distribs, 
                                cstart_time, transforms))

        weight_sumc += class_list[-1].weight
        if weight_sumd != 1.0:
            print 'Warning: weights sum for distribution of class %s is ' \
                  'not equal to one, weights ' \
                  'will be normalized.' % class_list[-1].label

    if weight_sumc != 1.0:
        print 'Warning: weights sum for the set of classes ' \
              'is not equal to one, weights will be normalized.'

    return class_list

def draw_cov_ellipse(centroid, cov_matrix, ax, 
                     perc=0.95, color='b'):
    """Draw the ellipse associated with the multivariate normal distribution
    defined by *centroid* and *cov_matrix*. The *perc* argument specified 
    the percentage of the distribution mass that will be drawn.

    This function is based on the example posted on Matplotlib mailing-list:
    http://www.mail-archive.com/matplotlib-users@lists.sourceforge.net/msg14153.html
    """
    U, s, _ = linalg.svd(cov_matrix)
    orient = atan2(U[1, 0], U[0, 0]) * 180.0/pi

    c = chi2.isf(1 - perc, len(centroid))
    width = 2.0 * sqrt(s[0] * c)
    height = 2.0 * sqrt(s[1] * c)

    ellipse = Ellipse(xy=centroid, width=width, height=height, 
                      angle=orient, fc=color, alpha=0.1)
    
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
    colors = [color_conv.to_rgba(COLORS[ref_labels.index(label)], 
                (i+1)*alph_inc) for i, label in enumerate(labels)]
    axis.scatter(x, y, c=colors, edgecolors='none')

    ellipses = []
    labels = []
    # Draw the distribution covariance ellipse
    for i, class_ in enumerate(class_list):
        present = False
        for distrib in class_.distributions:
            if time >= distrib.start_time:
                ref_ell = draw_cov_ellipse(distrib.centroid, distrib.matrix * distrib.scale,
                                           perc=0.95, ax=axis, color=COLORS[i])
                if not present:
                    ellipses.append(ref_ell)
                    labels.append(class_.label)
                    present = True

    axis.legend(ellipses, labels)
    fig.canvas.draw()

def weight_choice(seq):
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

def main(filename, samples, oracle, plot, path, seed=None):
    random.seed(seed)
    numpy.random.seed(seed)
    
    save = path is not None

    if (plot or save) and not MATPLOTLIB:
        print 'Warning: the --plot or --save-fig options were activated,'\
              'but matplotlib is unavailable. ' \
              'Processing will continue without plotting.'
    
    # Read file and initialize classes
    class_list = read_file(filename)
    
    # Initialize figure and axis before plotting
    if (plot or save) and MATPLOTLIB:
        fig = plt.figure(figsize=(10,10))
        ax1 = fig.add_subplot(111)
        if plot:
            plt.ion()
            plt.show()
        points = deque(maxlen=LAST_N_PTS)
        labels = deque(maxlen=LAST_N_PTS)
        ref_labels = map(attrgetter('label'), class_list)


    # Print CSV header
    if oracle:
        print "%s, %s, %s" % ('label',
                              ", ".join('x%i'% i for i in
                                        xrange(len(class_list[0].distributions[0].centroid))), 
                              ", ".join('%s' % class_.label for class_ in
                                        class_list))
    else:
        print "%s, %s" % ('label',
                          ", ".join('x%i'% i for i in
                                    xrange(len(class_list[0].distributions[0].centroid)))) 
        

    for i in xrange(samples):
        cclass = weight_choice([class_ for class_ in class_list 
                                if class_.start_time <= i])
        cdistrib = weight_choice([distrib for distrib in cclass.distributions
                                  if distrib.start_time <= i])
        spoint = cdistrib.sample()[0]

        # Compute the probability for each class
        # The probability of having sampled a point from a class C_i knowing x is given by :
        # P(C_i | x) = \frac{P(C_i) p(x | C_i)}{\sum_j{P(C_j) p(x | C_j) }}
        # p(x | C_i) = \sum_k{ P(G_k) p(x | G_k)}
        # Where G_k are the gaussian distribution of class C_i
        probs = []
        classes_weight_sum = sum(class_.weight for class_ in class_list 
                                 if class_.start_time <= i)
        for class_ in class_list:
            if class_.start_time <= i:
                prob_class = class_.weight / classes_weight_sum
                prob_dist = 0.0
                dists_weight_sum = sum(dist.weight 
                                       for dist in class_.distributions
                                       if dist.start_time <= i)
                for dist in class_.distributions:
                    prob_dist += dist.weight / dists_weight_sum * dist.pdf(spoint)
                probs.append(prob_class * prob_dist)
            else:
                probs.append(0.0)
        
        # Normalize probabilities
        probs = map(div, probs, repeat(sum(probs), len(probs)))

        # Print the sampled point in CSV format
        if oracle:
            print "%s, %s, %s" % (str(cclass.label), 
                                  ", ".join("%s" % v for v in spoint), 
                                  ", ".join("%.3f" % prob for prob in probs))
        else:
            print "%s, %s" % (str(cclass.label), 
                              ", ".join("%s" % v for v in spoint)) 

        # Plot the resulting distribution if required
        if (plot or save) and MATPLOTLIB:
            points.append(spoint)
            labels.append(class_.label)
            plot_class(i, ref_labels, class_list, points, labels, fig, ax1)
            if save:
                fig.savefig(path+'/point_%i.png' % i)
        
        # Update the classes' distributions
        for class_ in class_list:
            class_.update(i)

    if plot and MATPLOTLIB:
        plt.ioff()
        plt.show()

    return class_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read a file of classes'\
            'and return a series of randomly sampled points from those '\
            'classes.')
    parser.add_argument('filename', help='json file containing the classes')
    parser.add_argument('samples', type=int, help='number of samples')
    
    parser.add_argument('--oracle', dest='oracle', required=False, 
                        action='store_true', default=False,
                        help='append to the point its probability of ' \
                             'belonging to each class')
    parser.add_argument('--plot', dest='plot', required=False, 
                        action='store_true', default=False,
                        help='tell if the results should be plotted')
    parser.add_argument('--save-fig', dest='save_path', required=False, 
                        metavar='PATH', 
                        help='indicate where the figure should be saved')
    parser.add_argument('--seed', type=int, default=None, required=False, 
                        metavar='SEED',
                        help='random number generator seed')
    
    args = parser.parse_args()
    main(args.filename, args.samples, args.oracle, args.plot, args.save_path, args.seed)
