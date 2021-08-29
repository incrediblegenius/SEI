from sklearn import manifold
import matplotlib.pyplot as plt
import numpy as np
def Decto2d(X):
    tsne = manifold.TSNE(n_components=2, init='pca')
    X_tsne = tsne.fit_transform(X)
    return X_tsne

def Norm2Plot(X_tsne,targets):
    x_min, x_max = X_tsne.min(0) , X_tsne.max(0)
    X_norm = np.array((X_tsne - x_min) / (x_max - x_min))
    colors = np.array(['k','r','g','b','c','m','y','sienna','gold','lime',
                'midnightblue',
                'plum',
                'magenta',
                'navy',
                'coral',
                'olive',
                'seagreen',
                'teal',
                'dodgerblue',
                'firebrick',
                'salmon',
                'forestgreen',
                'crimson',
                'slateblue',
                'silver',
                'hotpink',
                'indianred',
                'indigo',
                'brown',
                'peru',
                'fuchsia',
                'cyan',
                'tan',
                'aqua',
                'mediumpurple'],dtype=str)
    plt.figure(figsize=(10,10))
    # for i in range(X_norm.shape[0]):
  
    plt.scatter(X_norm[:,0],X_norm[:,1],c = list(colors[np.array(targets)]),alpha=0.5)
    plt.legend(loc = 'best')
    plt.show()

def DecAndPlot(X,Y):
    Norm2Plot(Decto2d(X),Y)

