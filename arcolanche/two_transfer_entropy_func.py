from .utils import *
#from multiprocessing import Pool, cpu_count


def calc_two_te(three_tuple):
    x = three_tuple[0]
    y = three_tuple[1]
    z = three_tuple[2]

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

def shuffle_calc_two_te(three_tuple):
    y = three_tuple[1]
    z = three_tuple[2]

    y_rand_index = np.random.permutation(y.size)
    z_rand_index = np.random.permutation(z.size)

    y_shuffled = y[y_rand_index]
    z_shuffled = z[z_rand_index]
    
    shuffled_three_tuple = (three_tuple[0], y_shuffled, z_shuffled)
    
    res = calc_two_te(shuffled_three_tuple)
    
    return res
