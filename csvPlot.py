#!/usr/bin/env python
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as interpolate

##################################################
#   Utility program built on matplotlib library
#   to quickly plot files of the form:
#   x, y, err
#   from the command line
##################################################


# setup to take an argument of which file(s) the data is contained in
def get_user_args():
    """Read command line arguments and return as an argparse type""" 
    parser = argparse.ArgumentParser(description='Plot file(s) specified by the user. '+\
             'The format of the files should be space delimated columns containing the x, '+\
             'y and yerror data.')
    parser.add_argument('files', metavar='File', nargs='+', 
                    help='the name of the file(s) containing the data')
    parser.add_argument('--output', '-o', metavar='file', type=str,  
                    help='the name of the file to write to')
    parser.add_argument('-xb', metavar=('lower','upper'), type=float, 
                    help='the desired lower and upper bounds of the x-dimension', nargs=2)
    parser.add_argument('-yb', metavar=('lower','upper'), type=float, 
                    help='the desired lower and upper bounds of the y-dimension', nargs=2)
    parser.add_argument('--color', '-c', metavar='color', type=str, 
                    help='color list to plot the data points',nargs='+')
    parser.add_argument('--dimension', '-d', metavar=('width','height'), type=float, 
                    help='the figure dimensions in inches', nargs=2)
    parser.add_argument('--linewidth', '-l', metavar='width', type=float, 
                    help='the line width', nargs='+')
    parser.add_argument('-xl', metavar='xLabel', type=str, 
                    help='the label for the x-axis', nargs=1)
    parser.add_argument('-yl', metavar='yLabel', type=str, 
                    help='the label for the y-axis', nargs=1)
    parser.add_argument('-sx', metavar='xScale', type=float, 
                    help='the scaling factor for x-axis', nargs='+')
    parser.add_argument('-sy', metavar='yScale', type=float, 
                    help='the scaling factor for y-axis', nargs='+')
    parser.add_argument('--labels', metavar='label', type=str,
                    help='The labels for the figure inset', nargs='+' )
    parser.add_argument('--xtics', metavar='n', type=float, nargs=2,
                    help='The tic spacing. x major, x minor' )
    parser.add_argument('--ytics', metavar='n', type=float, nargs=2,
                    help='The tic spacing. y major, y minor' )
    parser.add_argument('--error', metavar='y/n', type=str, nargs='+',
                    help='Include error bars if present')
    parser.add_argument('--ecolor','-ec', metavar='color', type=str, nargs='+',
                    help='color for the error bars, if applicable' )
    parser.add_argument('--fit', metavar='n', type=float, nargs='+',
                    help='fit the data points with a spline with smoothing parameter n' )
    parser.add_argument('--logplot', metavar='x/y/b', type=str, nargs=1,
                    help='log plot of x, y, or both' )
    parser.add_argument('--invplot', metavar='x/y/b', type=str, nargs=1,
                    help='inverse plot of x, y, or both' )
    parser.add_argument('--linestyle', metavar='linestyle', type=str, nargs='+',
                    help='The linestyle for the plot' )
    parser.add_argument('--marker', metavar='marker', type=str, nargs='+',
                    help='The marker for the plot' )
    parser.add_argument('--markersize', metavar='markersize', type=float, nargs='+',
                    help='The marker size for the plot' )
    parser.add_argument('--hist', metavar='y/n', type=str, nargs=1,
                    help='make a histogram' )
    parser.add_argument('--nolabels', metavar='y/n', type=str, nargs=1,
                    help='dont show labels if y' )
    parser.add_argument('--legendsize', metavar='size', type=int, nargs=1,
                    help='scale the legend size' )
    parser.add_argument('--label2', metavar='a,b,...', type=str, nargs=1,
                    help='add a label to upper left to distinguish figures by a, b, etc')
    parser.add_argument('--label2size', metavar='num', type=float, nargs=1,
                    help='font size for label 2')

    return parser
##################################################

# read in data and return in a numpy array
def read_in_data(files, e=['n']*10):
    """Reads in two or three space separated columns from a data file.
        Returns a tuple of numpy arrays containing the x, y and error values.
        The latter is only included if the argument e='y'"""
    # initialize data
    data = []
    xs   = []
    ys   = []
    es   = []

    # read in files
    for f in files:
        with open(f) as F:
            data.append(F.read())
    # parse the data
    for i in range(len(data)):
        # split data and get rid of the last blank row and comments
        data[i] = data[i].split('\n')[:-1]
        data[i] = [ row for row in data[i] if row[0] != "@" ]
        data[i] = [ row for row in data[i] if row[0] != "#" ]

        # save in x and y variables
        data[i] = [ row.replace(',',' ').split() for row in data[i] ]

        xs.append( np.asarray([float(row[0]) for row in data[i]]) )
        ys.append( np.asarray([float(row[1]) for row in data[i]]) )
        if e[i] == 'y':
            try:
                es.append( np.asarray([ float(row[2]) for row in data[i] ]) )
            except:
                # if no error column provided, just set to zero
                es.append( np.asarray([0.]*len(data[i])) )
        else:
            es.append( np.asarray([0.]*len(data[i])) )

    return xs, ys, es

##################################################
       
def scale_data( x, s ):
    for i in range(len(x)):
        x[i] *= s[i]
    return x

##################################################

# The main routine
# get the user input
args = get_user_args()

# store the user arguments in variables
files     = args.parse_args().files
errplt    = args.parse_args().error if args.parse_args().error else ['n']*len(files)
histplt   = args.parse_args().hist[0] if args.parse_args().hist else 'n'
linewidth = args.parse_args().linewidth if args.parse_args().linewidth else [1]*len(files)
linestyle = args.parse_args().linestyle if args.parse_args().linestyle else ['-']*len(files)
marker    = args.parse_args().marker if args.parse_args().marker else [' ']*len(files)
markersize= args.parse_args().markersize if args.parse_args().markersize else [3.5]*len(files)
output    = args.parse_args().output if args.parse_args().output else None
width     = args.parse_args().dimension[0] if args.parse_args().dimension else 3.3
height    = args.parse_args().dimension[1] if args.parse_args().dimension else 2.5
dpi       = 300
xb        = args.parse_args().xb
yb        = args.parse_args().yb
color     = args.parse_args().color
nolabels  = args.parse_args().nolabels[0] if args.parse_args().nolabels else 'n'
label2    = args.parse_args().label2 if args.parse_args().label2 else None
label2size= args.parse_args().label2size if args.parse_args().label2size else 10

if not color: # default color
    color = ['black','red','blue','green','purple','magenta','cyan','orange','yellow'][0:len(files)]
ecolor    = args.parse_args().ecolor
if not ecolor: # default color
    ecolor = ['magenta','blue','cyan','green','orange','red'][0:len(files)]
fit       = args.parse_args().fit if args.parse_args().fit else [-1]*len(files)
labels    = args.parse_args().labels
if not labels:  # default labels 0, 1, 2, 3 if not given by user
    labels = range(len(files))
xlabels   = args.parse_args().xl
if not xlabels: # default x label
    xlabels = ['x']
ylabels   = args.parse_args().yl
if not ylabels: #
    ylabels = ['y']
sx        = args.parse_args().sx
sy        = args.parse_args().sy
xtics     = args.parse_args().xtics
ytics     = args.parse_args().ytics
logplot   = args.parse_args().logplot[0] if args.parse_args().logplot else 'n'
invplot   = args.parse_args().invplot[0] if args.parse_args().invplot else 'n'
legendsize= args.parse_args().legendsize[0] if args.parse_args().legendsize else None

# read the files and store data
x, y, e = read_in_data( files, e=errplt )

# if an inverse plot is requested, invert the appropriate data
if invplot == 'x' or invplot == 'b':
    x = 1./x
if invplot == 'y' or invplot == 'b':
    y = 1./y
    e = 1./e

# scale the data if requested by the user
if sx:
    x = scale_data( x, sx )
if sy:
    y = scale_data( y, sy )
    e = scale_data( e, sy )

##################################################
# make the plots
plt.figure(1, figsize=(width, height), dpi=dpi )
for i in range(len(y)):
    # error plot is a little different from regular
    if errplt[i] =='y':
        plt.errorbar( x[i], y[i], yerr=e[i], label=labels[i], linewidth=linewidth[i], 
                linestyle=linestyle[i], color=color[i], fmt=marker[i], capthick=linewidth[i], 
                barsabove=False, elinewidth=linewidth[i], ecolor=ecolor[i], 
                markersize=markersize[i], capsize=2.5 )
    elif histplt =='y':
        plt.bar( x[i], y[i], color=color[i], edgecolor=ecolor[i], align='center', 
                 width=abs(x[i][0]-x[i][1]), label=labels[i] )
    else:
        plt.plot( x[i], y[i], label=labels[i], linewidth=linewidth[i], linestyle=linestyle[i],
                color=color[i], marker=marker[i], markersize=markersize[i] )
    if fit[i] >= 0:
        # order, must be for spline to work
        order = np.argsort(x[i])
        spl = interpolate.UnivariateSpline(x[i][order],y[i][order], s=fit[i])
        xs = np.linspace( x[i].min(), x[i].max(), num=100 )
        plt.plot( xs, spl(xs), linewidth=linewidth[i], color=color[i], linestyle='-' )

# edit the plots with the appropriate labels and stuff

plt.xlabel( xlabels[0], fontsize=10 )
plt.ylabel( ylabels[0], fontsize=10)

# adjust the tick size and turn on minor ticks
plt.tick_params(axis='both',which='major',labelsize=10, direction='in',
                top='on',bottom='on', left='on', right='on')
plt.minorticks_on()
plt.tick_params(axis='both',which='minor',labelsize=6, direction='in',
                top='on',bottom='on', left='on', right='on')

# if log plots are requested
if logplot == 'x' or logplot == 'b':
    plt.xscale('log')
if logplot == 'y' or logplot == 'b':
    plt.yscale('log')

# set the tics if provided by the user, else use the default
from matplotlib.ticker import MultipleLocator
if xtics:
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(xtics[0]))
    ax.xaxis.set_minor_locator(MultipleLocator(xtics[1]))
if ytics:
    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(ytics[0]))
    ax.yaxis.set_minor_locator(MultipleLocator(ytics[1]))



# adjust the area to the left and right of the plot to fit the labels
plt.subplots_adjust(bottom=0.2,left=0.20)
# plot the curve legend
if nolabels != 'y':
    if legendsize:
        plt.legend(prop={'size':legendsize},frameon=False)
    else:
        plt.legend(prop={'size':10},frameon=False)

# adjust window if user has added bounds
if xb:
   plt.xlim((xb[0],xb[1])) 
if yb:
   plt.ylim((yb[0],yb[1])) 

# add second label, as a, b, c if desired
if label2:
    ax = plt.gca()
    ax.text(0.05, 0.95, label2[0], fontsize=label2size, transform=ax.transAxes, va='top')#, fontweight='bold' )


# show the figure if no output file is specified, otherwise save it
if output == None:
    plt.show()
    plt.close()
else:
    plt.savefig(output, dpi=dpi)

# exit
exit()
