import matplotlib.pyplot as plt

def plotOne(path, points):

    """
    path: List of lists with the different orders in which the nodes are visited
    points: coordinates for the different nodes
    num_iters: number of paths that are in the path list
    
    """

    # Unpack the primary TSP path and transform it into a list of ordered 
    # coordinates

    x = points[:,0]
    y = points[:,1]

    plt.plot(x, y, 'co')

    # Set a scale for the arrow heads (there should be a reasonable default for this, WTF?)
    a_scale = float(max(x))/float(100)

    # Draw the primary path for the TSP problem
    plt.arrow(x[-1], y[-1], (x[0] - x[-1]), (y[0] - y[-1]), head_width = a_scale, 
            color ='g', length_includes_head=True)
    for i in range(0,len(x)-1):
        plt.arrow(x[i], y[i], (x[i+1] - x[i]), (y[i+1] - y[i]), head_width = a_scale,
                color = 'g', length_includes_head = True)

def plotAll(paths, names):

    nb_graphs = len(paths)
    tsp_len = len(paths[0])
    ordering=list(range(0,tsp_len))
    f = plt.figure(figsize=(15,5))
    for i, (path,name) in enumerate(zip(paths,names)):
        f.add_subplot(1,nb_graphs,i+1,aspect='equal')
        plt.title(name)
        plotOne(ordering,path)
    
    plt.show()