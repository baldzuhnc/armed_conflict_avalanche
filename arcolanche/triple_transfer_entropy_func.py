from .utils import *
#from multiprocessing import Pool, cpu_count



def iter_polygon_triple(polygon_triple, number_of_shuffles, time_series,
                      n_cpu=None):
    
    """
    Returns
    -------
    dict
        Key is directed edge between two polygons Y,Z and a polygon X. Value is list (TE, TE shuffles).
    """

    # calculate transfer entropy for a single pair of tiles
    def loop_wrapper(triple):
        
        #unpack triple of polygon ids
        x = time_series[triple[0]].values
        y = time_series[triple[1]].values
        z = time_series[triple[2]].values
        
        #calculate te and shuffles
        return triple, (calc_two_te(x,y,z),
                      [shuffle_calc_two_te(x,y,z) for i in range(number_of_shuffles)])
    
    if n_cpu is None:
        triple_te = {}
        for triple in polygon_triple:
            triple_te[triple] = loop_wrapper(triple)[1]
    else:
        # note that each job is quite fast, so large chunks help w/ speed
        with Pool() as pool:
            triple_te = dict(pool.map(loop_wrapper, polygon_triple, chunksize=200))
    return triple_te

def calc_two_te(x,y,z):
    xt = x[1:]
    xt1 = x[:-1]

    yt = y[1:] 
    zt = z[1:]

    #Step 1: Joint distribution P(zt, yt, xt, xt1)
    joint = np.zeros((2,2,2,2))
    joint[0][0][0][0] = ((zt == 0) & (yt == 0) & (xt == 0) & (xt1 == 0)).mean()
    joint[0][0][0][1] = ((zt == 0) & (yt == 0) & (xt == 0) & (xt1 == 1)).mean()
    joint[0][0][1][0] = ((zt == 0) & (yt == 0) & (xt == 1) & (xt1 == 0)).mean()
    joint[0][0][1][1] = ((zt == 0) & (yt == 0) & (xt == 1) & (xt1 == 1)).mean()

    joint[0][1][0][0] = ((zt == 0) & (yt == 1) & (xt == 0) & (xt1 == 0)).mean()
    joint[0][1][0][1] = ((zt == 0) & (yt == 1) & (xt == 0) & (xt1 == 1)).mean()
    joint[0][1][1][0] = ((zt == 0) & (yt == 1) & (xt == 1) & (xt1 == 0)).mean()
    joint[0][1][1][1] = ((zt == 0) & (yt == 1) & (xt == 1) & (xt1 == 1)).mean()

    joint[1][0][0][0] = ((zt == 1) & (yt == 0) & (xt == 0) & (xt1 == 0)).mean()
    joint[1][0][0][1] = ((zt == 1) & (yt == 0) & (xt == 0) & (xt1 == 1)).mean()
    joint[1][0][1][0] = ((zt == 1) & (yt == 0) & (xt == 1) & (xt1 == 0)).mean()
    joint[1][0][1][1] = ((zt == 1) & (yt == 0) & (xt == 1) & (xt1 == 1)).mean()

    joint[1][1][0][0] = ((zt == 1) & (yt == 1) & (xt == 0) & (xt1 == 0)).mean()
    joint[1][1][0][1] = ((zt == 1) & (yt == 1) & (xt == 0) & (xt1 == 1)).mean()
    joint[1][1][1][0] = ((zt == 1) & (yt == 1) & (xt == 1) & (xt1 == 0)).mean()
    joint[1][1][1][1] = ((zt == 1) & (yt == 1) & (xt == 1) & (xt1 == 1)).mean()

    #create named dataframe for joint
    #joint_df = pd.DataFrame(joint.reshape(-1), columns = ['P'])
    #joint_df.index = pd.MultiIndex.from_product([['zt=0', 'zt=1'], ['yt=0', 'yt=1'], ['xt=0', 'xt=1'], ['xt1=0', 'xt1=1']], names = ['zt', 'yt', 'xt', 'xt1'])
    #joint_df

    #Step 2: Marginal distributions
    p_xt = joint.sum(axis = (0,1,3)) #sum over yt, zt and xt1
    p_xt_xt1 = joint.sum(axis = (0,1)) #sum over yt, zt 
    p_zt_yt_xt = joint.sum(axis = 3) #sum over xt1

    #Step 3: Conditional distributions
    p_xt1_given_xt = p_xt_xt1/p_xt[:, np.newaxis] #p(xt1|xt) = p(xt1, xt)/p(xt) #2x2, add axis for explicit broadasting: colwise!, axes are (xt, xt1)

    p_xt1_given_zt_yt_xt = joint/p_zt_yt_xt[:, :, :, np.newaxis] #p(xt1|zt, yt, xt) = p(xt1, zt, yt, xt)/p(zt, yt, xt), add axis for explicit broadasting: colwise!

    #Step 4: Transfer Entropy
    np.seterr(divide='ignore', invalid='ignore')
    te = np.nansum(joint * np.log2(p_xt1_given_zt_yt_xt/p_xt1_given_xt[np.newaxis, np.newaxis, :, :])) #calculate sum only for terms which dont include log(0)
    
    return te

def shuffle_calc_two_te(x,y,z):

    y_rand_index = np.random.permutation(y.size)
    z_rand_index = np.random.permutation(z.size)

    y_shuffled = y[y_rand_index]
    z_shuffled = z[z_rand_index]
    
    res = calc_two_te(x, y_shuffled, z_shuffled)
    
    return res
