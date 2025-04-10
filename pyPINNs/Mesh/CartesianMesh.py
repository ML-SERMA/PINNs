import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import copy

class CartesianMesh:
    def __init__(self, ndim=2, xmin=[0, 0], xmax = [1, 1], nmail=[2, 2]):
        self.ndim = ndim
        self.xmin = xmin
        self.xmax = xmax
        self.nmail = nmail
        #self.femOrder = femOrder
        self.nCell = np.prod(nmail)
        self.generateMesh()


    def generateMesh(self):
        #self.pas = [np.ones(self.nmail[idim]) * (self.xmax[idim] - self.xmin[idim]) / self.nmail[idim] for idim in range(self.ndim)]
        self.pas = [[float((self.xmax[idim] - self.xmin[idim])* 1.0/self.nmail[idim]) for _ in range(self.nmail[idim])] for idim in range(self.ndim)]

        self.xpos = [[] for i in range(self.ndim)]
        for idim in range(self.ndim):
            self.xpos[idim].append(self.xmin[idim])
            for iCell in range(self.nmail[idim]):
                self.xpos[idim].append( self.xpos[idim][iCell] + self.pas[idim][iCell] )
            self.xpos[idim][self.nmail[idim]] = self.xmax[idim]


        self.xmid = [[(self.xpos[idim][iCell] + self.xpos[idim][iCell + 1])*0.5 for iCell in range(self.nmail[idim])]
                        for idim in range(self.ndim)]


    def refineMesh(self, vnsPas):
        copynmail = copy.deepcopy(self.nmail)
        for idim in range(self.ndim):
            output = []
            for iCell in range(copynmail[idim]):
                if vnsPas[idim][iCell] > 1:
                    self.nmail[idim] += (vnsPas[idim][iCell] - 1)
                    Replace = [(self.pas[idim][iCell])*1.0/vnsPas[idim][iCell] for _ in range(vnsPas[idim][iCell])]
                    output.extend(Replace)
                else:
                    output.extend([self.pas[idim][iCell]])
            self.pas[idim] = output
            #print(self.pas[idim])
        self.nCell = np.prod(self.nmail)
        self.xpos = [[] for i in range(self.ndim)]
        for idim in range(self.ndim):
            self.xpos[idim].append(self.xmin[idim])
            for iCell in range(self.nmail[idim]):
                self.xpos[idim].append(self.xpos[idim][-1] + self.pas[idim][iCell])
        self.xmid = [[(self.xpos[idim][iCell] + self.xpos[idim][iCell + 1])*0.5 for iCell in range(self.nmail[idim])] for idim in range(self.ndim)]
    def modifyMesh(self,xpos):
        
        self.xpos = xpos
        self.ndim = len(self.xpos)
        self.xmin = [ self.xpos[idim][0] for idim in range(self.ndim)]
        self.xmax = [ self.xpos[idim][-1] for idim in range(self.ndim)]
        self.nmail = [ len(self.xpos[idim]) -1 for idim in range(self.ndim)]
        self.nCell = np.prod(self.nmail)
        self.pas = [ list(np.array(self.xpos[idim])[1:] - np.array(self.xpos[idim])[0:-1])   for idim in range(self.ndim)] 
        self.xmid = [ list((np.array(self.xpos[idim])[1:] + np.array(self.xpos[idim])[0:-1])*0.5)  for idim in range(self.ndim)] 

    def plotMesh(self):
        if self.ndim == 2:
            x = np.asarray(self.xpos[0])
            y = np.asarray(self.xpos[1])
            xv, yv = np.meshgrid(x, y)
            plt.plot(xv, yv, marker='.', color='k', linestyle='none')
            plt.show()


    def viewGrid(self):
        if self.ndim == 2:
            xs = np.asarray(self.xpos[0])
            ys = np.asarray(self.xpos[1])
            ax = plt.gca()
            # grid "shades" (boxes)
            w, h = xs[1] - xs[0], ys[1] - ys[0]
            for i, x in enumerate(xs[:-1]):
                for j, y in enumerate(ys[:-1]):
                    #if i % 2 == j % 2:  # racing flag style
                    ax.add_patch(Rectangle((x, y), w, h, fill=True, color='#008610', alpha=.2))
            # grid lines
            for x in xs:
                plt.plot([x, x], [ys[0], ys[-1]], color='black', alpha=.33, linestyle='-')
            for y in ys:
                plt.plot([xs[0], xs[-1]], [y, y], color='black', alpha=.33, linestyle='-')
            plt.show()




