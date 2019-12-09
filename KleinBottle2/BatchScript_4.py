#https://stackoverflow.com/questions/20961287/what-is-pylab
#instead of %pylab inline as a magic function, we need to import each incarnation, check the link above.
'''Step 1 - Getting the point cloud and Computing Vietoris-Rips Complexes and Cohomology'''
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
#plt is the matplotlib incarnation.
import examples as eg
import numpy as np
from numpy import *
import dionysus
#set up random seed in numpy for reproducibility.
np.random.seed(seed=1234)
'''
Setting for iterations for optimization and sample size of points, L^p, L^q array, weight alpha array
'''
iter=10001
Nsample=200
lp=2
lq_array=np.array([1,1.5,1.8,2,2.25,3,512])
alpha_array=np.array([1,0.975,0.95,0.90,0.7,0.5,0.3,0.1,0.05,0.025,0])
counter=0#image counter

#points = eg.double_pinched_torus_example(n=200,a=.2,b=.2,c=2,d=.1)
points = eg.klein_bottle_example_4D(n=Nsample)
#Now we can plot the points using hue to show the Z coordinate
plt.scatter(points[:,0], points[:,1], c = points[:,2], cmap = 'jet')
plt.clim(-.1, .1)
plt.colorbar()
plt.axis('equal')
plt.title('Visualizing Z coordinates')
plt.figure()
#The examples.py generates data points in form of point clouds that can be analyzed using the imported dionysus module.
prime = 23
#choose the prime base for the coefficient field that we use to construct the persistence cohomology.
vr = dionysus.fill_rips(points, 2, 4.) #Vietoris-Rips complex up to dimension 2 and maximal distance 4
cp = dionysus.cohomology_persistence(vr, prime, True) #Create the persistent cohomology based on the chosen parameters.
dgms = dionysus.init_diagrams(cp, vr) #Calculate the persistent diagram using the designated coefficient field and complex.
dionysus.plot.plot_bars(dgms[1], show=True)
dionysus.plot.plot_diagram(dgms[1], show=True)
#dionysus.plot.plot_diagram(dgms[0], show=True)
#Plot the barcode and diagrams using matplotlib incarnation within Dionysus2. This mechanism is different in Dionysus.
'''Step 2 - Selecting the cocycle and visualization.'''
persistence_threshold = 1
bars = [bar for bar in dgms[1] if bar.death-bar.birth > persistence_threshold]
#choosing cocycle that persist at least threshold=1.
cocycles = [cp.cocycle(bar.data) for bar in bars]
#Red highlight cocyles that persist more than threshold value on barcode, when more than one cocyles have persisted over threshold values, this plots the first one.
dionysus.plot.plot_bars(dgms[1], show=False)
plt.plot([[bar.birth,bar.death] for bar in dgms[1] if bar.death-bar.birth > persistence_threshold][0],[[x,x] for x,bar in enumerate(dgms[1]) if bar.death-bar.birth > persistence_threshold][0],'r')
plt.title('Showing the selected cycles on bar codes (red bars)')
plt.show()
#Red highlight ***ALL*** cocyles that persist more than threshold value on diagram.
dionysus.plot.plot_diagram(dgms[1], show=False)
Lt1 = [[point.birth,point.death] for point in dgms[1] if point.death-point.birth > persistence_threshold]
for Lt3 in Lt1:
    print(Lt3)
    plt.plot(Lt3[0],Lt3[1],'ro')
plt.title('Showing the selected cycles on diagram (red points)')
plt.show()
chosen_cocycle= cocycles[0]
chosen_bar= bars[0]
#This is the same as choosing the maximal persistent cocycle when there is only one candidate cocycle.
#print(chosen_cocycle)
print(chosen_bar)
'''Step 3 - Display the scatter points sampled from the manifold'''
pt = max(dgms[1], key = lambda pt: pt.death - pt.birth)
#print(pt)
chosen_cocycle = cp.cocycle(pt.data)
chosen_bar     = [bar for bar in dgms[1] if bar.death==pt.death and bar.birth==pt.birth]
chosen_bar     = chosen_bar[0]
#print(chosen_cocycle)
print(chosen_bar)
#fill_rips() computes Vietoris–Rips filtrations (up to a specified skeleton dimension and distance r).
#If it is computed the smoothed coefficients can be used as initial condition for the optimization code
#hrluo: However, this optimization problem seems extremely sensitive to the initialization, the choice of L^2 smoothed coordinates does not seem to be
vr_complex = dionysus.Filtration([s for s in vr if s.data <= max([bar.birth for bar in bars])])
coords = dionysus.smooth(vr_complex, chosen_cocycle, prime)
##To smooth the cocycle and convert it to the corresponding *circular coordinates*, we need to choose a complex,
##in which we do the smoothing.
##Here we select the complex in the filtration that exists at the midvalue of the persistence bar, (pt.death + pt.birth)/2:
##or the complex with the maximal birth
#https://mrzv.org/software/dionysus2/tutorial/cohomology.html
toll = 1e-5
p,val = (chosen_bar,coords)
#Show the constant edges first.
edges_costant = []
thr = p.birth # i want to check all edges that were there when the cycle was created
for s in vr:
    if s.dimension() != 1:
        continue
        #Only want edges in dim 1.
    elif s.data > thr:
        #print(s.data)
        #Only want those edges that exist when the chosen_bar is born.
        break
    if abs(val[s[0]]-val[s[1]]) <= toll:
        edges_costant.append([points[s[0],:],points[s[1],:]])
edges_costant = np.array(edges_costant)
plt.plot(edges_costant.T[0,:],edges_costant.T[1,:], c='k')
#Now we can plot the points using hue to show the circular coordinate
plt.scatter(points[:,0], points[:,1], c = coords, cmap = 'jet')
plt.clim(-.1, .1)
plt.colorbar()
plt.axis('equal')
plt.title('Visualizing constant edges')
plt.show()
'''Step 4 - Second smoothing using a new cost function'''
import utils
l2_cocycle,f,bdry = utils.optimizer_inputs(vr, bars, chosen_cocycle, coords, prime)
#l2_cocycle = l2_cocycle.reshape(-1, 1)
#np.zeros(l2_cocycle.shape[0])
#l2_cocycle.shape
#f-bdry*l2_cocycle
ZV = np.zeros(l2_cocycle.shape[0])
ZV = ZV.reshape(l2_cocycle.shape[0],1)
ZV.shape
def plotCir(res,keywrd='TensorFlow',ctr=0):
    color = np.mod(res.T[0,:],1)
    toll = 1e-5#tolerance for constant edges
    edges_constant = []
    thr = chosen_bar.birth # i want to check constant edges in all edges that were there when the cycle was created
    for s in vr:
        if s.dimension() != 1:
            continue
        elif s.data > thr:
            break
        if abs(color[s[0]]-color[s[1]]) <= toll:
            edges_constant.append([points[s[0],:],points[s[1],:]])
    edges_constant = np.array(edges_constant)
    #scatter(*points.T, c=color, cmap="hsv", alpha=.5)
    plt.clf()
    plt.scatter(points.T[0,:],points.T[1,:],s=20, c=color, cmap="jet")
    plt.clim(-1, 1)
    plt.colorbar()
    plt.axis('equal')
    plt.title('Z_{0:d} coefficient smoothed values mod 1 \n {1:.2f}*L{2:.2f} + {3:.2f}*L{4:.2f} norm ({5})'.format(prime,1-alpha,lp,alpha,lq,keywrd))
    #plot(*edges_constant.T, c='k')
    if edges_constant.shape[0]>0:
        plt.plot(edges_constant.T[0,:],edges_constant.T[1,:], c='k', alpha=1)
    plt.savefig('PinchedTorusPlot_Z{}_{}_{}.png'.format(prime,keywrd,ctr))
    #plt.show()
import pkg_resources
pkg_resources.require("tensorflow==1.15")
import tensorflow as tf
#Following seems working, c.f.
#https://stackoverflow.com/questions/55552715/tensorflow-2-0-minimize-a-simple-function
#scipy.sparse.csr.csr_matrix
B_mat = bdry.todense()
#print(f.shape)
#print((B_mat*l2_cocycle).shape)
###L1 in tensorflow language
#cost_z = tf.reduce_sum( tf.abs(f - B_mat @ z) )
###L2 in tensorflow language
#cost_z = tf.reduce_sum( tf.pow( tf.abs(f - B_mat @ z),2 ) )
#Lp+alpha*Lq norm in tensorflow language
total=lq_array.shape[0]*alpha_array.shape[0]
for lq in lq_array:
    for alpha in alpha_array:
        counter=counter+1
        print(counter,'/',total)
        print('(',1-alpha,')L^',lp,'+(',alpha,')L^',lq,'\n')
        #print(lq)
        #alpha=1
        #cost_z = (1-alpha)*tf.pow( tf.reduce_sum( tf.pow( tf.abs(f - B_mat @ z),lp ) ), 1/lp) + alpha* tf.pow( tf.reduce_sum( tf.pow( tf.abs(f - B_mat @ z),lq ) ), 1/lq)
        init_val=ZV
        ##########
        #Gradient Descedent Optimizer
        z = tf.Variable(initial_value=init_val, name='z')
        opt = tf.train.GradientDescentOptimizer(0.1)
        cost_z = (1-alpha)*tf.pow( tf.reduce_sum( tf.pow( tf.abs(f - B_mat @ z),lp ) ), 1/lp) + alpha* tf.pow( tf.reduce_sum( tf.pow( tf.abs(f - B_mat @ z),lq ) ), 1/lq)
        train = opt.minimize(cost_z)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for step in range(iter):
            sess.run(train)
            if step % 500 == 0:
                print('GD',step, sess.run(cost_z))

        res_gd=sess.run(z)
        ##########

        ##########
        #Adams Optimizer
        z = tf.Variable(initial_value=init_val, name='z')
        opt = tf.train.AdamOptimizer(0.1)
        cost_z = (1-alpha)*tf.pow( tf.reduce_sum( tf.pow( tf.abs(f - B_mat @ z),lp ) ), 1/lp) + alpha* tf.pow( tf.reduce_sum( tf.pow( tf.abs(f - B_mat @ z),lq ) ), 1/lq)
        train = opt.minimize(cost_z)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for step in range(iter):
            sess.run(train)
            if step % 500 == 0:
                print('Adam',step, sess.run(cost_z))

        res_adam=sess.run(z)
        ##########

        ##########
        #AdagradOptimizer
        z = tf.Variable(initial_value=init_val, name='z')
        opt = tf.train.AdagradOptimizer(0.1)
        cost_z = (1-alpha)*tf.pow( tf.reduce_sum( tf.pow( tf.abs(f - B_mat @ z),lp ) ), 1/lp) + alpha* tf.pow( tf.reduce_sum( tf.pow( tf.abs(f - B_mat @ z),lq ) ), 1/lq)
        train = opt.minimize(cost_z)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for step in range(iter):
            sess.run(train)
            if step % 500 == 0:
                print('Adagrad',step, sess.run(cost_z))

        res_adagrad=sess.run(z)
        ##########
        print(np.ptp(res_gd,axis=0))
        print(np.ptp(res_adam,axis=0))
        print(np.ptp(res_adagrad,axis=0))

        from datetime import datetime
        now = datetime.now() # current date and time
        timestr = now.strftime("%H:%M:%S")

        plotCir(res_gd,timestr+'KleinBottle4D_gradient',ctr=counter)
        plotCir(res_adam,timestr+'KleinBottle4D_adams',ctr=counter)
        plotCir(res_adagrad,timestr+'KleinBottle4D_adagrad',ctr=counter)
