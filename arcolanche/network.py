# ====================================================================================== #
# Causal avalanche network methods using transfer entropy.
# Author : Eddie Lee, Niraj Kushwaha
# ====================================================================================== #
import networkx as nx
from scipy.sparse import lil_matrix, csr_matrix
from .transfer_entropy_func import *
from .self_loop_entropy_func import *

from .utils import *


def links(time_series, neighbor_info_dataframe,
          number_of_shuffles, degree):
    """Calculates transfer entropy and identifies significant links between Voronoi
    neighbors assuming a 95% confidence interval.

    Parameters
    ----------
    time_series : pd.DataFrame
    neighbor_info_dataframe : pd.DataFrame
    number_of_shuffles : int
    
    Returns
    -------
    dict
        dict with keys as directed edge and
        values as tuple where first element is
        the transfer entropy and the second
        element is a list of shuffled transfer entropies.
    """

    def get_tuples():
        #add argument for time series check, or do later
        # Initialize an empty sparse adjacency matrix of nxn
        n = len(neighbor_info_dataframe)

        adjacency_matrix = lil_matrix((n, n), dtype=int) #list of lists format

        cell_id_to_position = {cell_id: pos for pos, cell_id in enumerate(neighbor_info_dataframe.index)}

        #iterate over (column name, series), fill adjacency matrix
        for cell_id, neighbors in neighbor_info_dataframe['neighbors'].items():
            #current cell_id has to be in time_series (not in ts if zero activity)
            if cell_id in time_series.columns:
                for neighbor in neighbors:
                    #neighbour has to be in polygons and time series
                    if neighbor in cell_id_to_position and neighbor in time_series.columns: 
                        #at position of cell_id and neighbor, set value to 1 -> iteratively fill adjacency matrix
                        adjacency_matrix[cell_id_to_position[cell_id], cell_id_to_position[neighbor]] = 1

        # Convert the adjacency matrix to CSR format for efficient matmul (also done implicitly)
        adjacency_matrix = adjacency_matrix.tocsr()
        adjacency_matrix_power = adjacency_matrix.copy()

        
        #fill connections dict first. keys: (cell1, cell2) values = degree of connection
        connections = {}

        for d in range(1, degree+1):
            if d > 1:
                adjacency_matrix_power = adjacency_matrix_power @ adjacency_matrix

            #remove self loops
            adjacency_matrix_power.setdiag(0)
            #get indices of non-zero values
            rows, cols = adjacency_matrix_power.nonzero()

            for row, col in zip(rows, cols):
                #add to connections if not already in
                if (row, col) not in connections:
                    connections[(row, col)] = d

        #return tuples of (cell1, cell2, first degree of connection)
        tuples = [(neighbor_info_dataframe.index[row], neighbor_info_dataframe.index[col], d) for (row, col), d in connections.items()]
        
        #if adj:
        #    adjacency_df_power = pd.DataFrame(adjacency_matrix_power.toarray(), index=neighbor_info_dataframe.index, columns=neighbor_info_dataframe.index)
        #    return tuples, adjacency_df_power
        
        return tuples


    pair_poly_te = iter_polygon_pair(get_tuples(),
                                    number_of_shuffles, 
                                    time_series)
    return pair_poly_te


def self_links(time_series, number_of_shuffles):
    """Calculates self loop transfer entropy and identifies polygons with significant
    self loops assuming a 95% confidence interval.

    Parameters
    ----------
    time_series : pd.DataFrame
    number_of_shuffles : int
    
    Returns
    -------
    dict
        dict with keys as self loop tiles and
        values as tuple where first element is
        the self transfer entropy and the second
        element is a list of shuffled self transfer entropies.
    """  
    
    def valid_polygons_finder():
        valid_polygons = time_series.columns.astype(int).to_list()

        return valid_polygons

    valid_poly_te = iter_valid_polygons(valid_polygons_finder(),
                                        number_of_shuffles,
                                        time_series)

    return valid_poly_te


class CausalGraph(nx.DiGraph):
    def setup(self, self_poly_te, pair_poly_te, sig_threshold=95):
        """
        Parameters
        ----------
        self_poly_te : dict
            Keys are twoples. Values are TE and TE shuffles.
        pair_poly_te : dict
            Keys are twoples. Values are TE and TE shuffles.
        sig_threshold : float, 95
        """

        assert 0<=sig_threshold<=100 and isinstance(sig_threshold, int)
        self.self_poly_te = self_poly_te
        self.pair_poly_te = pair_poly_te
        self.sig_threshold = sig_threshold

        self.build_causal()
        
    def build_causal(self):
        """Build causal network using self.sig_threshold.
        """

        for poly, (te, te_shuffle) in self.self_poly_te.items():
            if (te>te_shuffle).mean() >= (self.sig_threshold/100):
                self.add_edge(poly, poly, weight=te)
            else:
                self.add_node(poly)

        for pair, (te, te_shuffle) in self.pair_poly_te.items():
            if (te>te_shuffle).mean() >= (self.sig_threshold/100):
                self.add_edge(pair[0], pair[1], weight=te)

        self.uG = self.to_undirected()

    def self_loop_list(self,info=False):
        """Outputs a list of all nodes which have a self loop.

        Returns
        -------
        list
            A list of all nodes which have a self loop.
        """

        self_loop_node_list = []
        for i in self.edges(data=info):
            if(i[0] == i[1]):
                if(info):
                    self_loop_node_list.append((i[0],i[2]))
                else:
                    self_loop_node_list.append(i[0])

        return self_loop_node_list


    def edges_no_self(self,info=False):
        """Outputs a list of tuples where each tuple contains node index which has a
         causal link between them.
        info: bool , False
            The edge attribute returned in 3-tuple (u, v, ddict[data]). If True, 
            return edge attribute dict in 3-tuple (u, v, ddict). If False, return 2-tuple (u, v).
        Returns
        -------
        list
            A list of tuples where each tuple contain two nodes which have a link
            between them. 
        """
        return [i for i in self.edges(data=info) if i[0] != i[1]]


    def causal_neighbors(self):
        """Outputs a dict where keys are node index and values are list of
         successive neighbors.
        """
        neighbor_dict = {}
        for node in self.nodes:
            neighbor_list_temp = []
            for neighbor in self.successors(node):
                if(node != neighbor):
                    neighbor_list_temp.append(neighbor)
                neighbor_dict[node] = neighbor_list_temp

        return neighbor_dict
# end CausalGraph    
