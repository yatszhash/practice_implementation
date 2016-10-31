from itertools import chain

from nlp_ml.seq_labeling.Edge import Edge
from nlp_ml.seq_labeling.Node import *


class FBAGraph:
    '''
        for training of CRF
    '''

    def __init__(self):
        self.node_list = []
        self.edge_lst = []
        self.z_coef = -1
        self.X = None

    def set_nodes(self):
        pass

    def add_nodes(self, node):
        self.node_list.append(node)

    def initialize(self, X, labels):
        self.X = X
        X_size = len(X)
        self.node_list.append([Node("", "BOS", 0)])
        inner_layer = [self.nodes_from_label_set(word, labels, layer + 1)
                       for layer, word in enumerate(X)]
        self.node_list.extend(inner_layer)
        self.node_list.append([Node("", "EOS", X_size + 1)])

        for idx, layer_nodes in enumerate(self.node_list):
            for node in layer_nodes:
                if idx > 0:
                    for prev_node in self.node_list[idx - 1]:
                        edge = Edge(prev_node, node, idx)
                        self.edge_lst.append(edge)
                        node.add_prev_nodes(prev_node, edge)
                        prev_node.add_next_nodes(node, edge)

    def nodes_from_label_set(self, word, labels, layer):
        return [Node(word, label, layer) for label in labels]

    def search_edge(self, prev_node_label, node_label, layer):
        return list(filter(lambda edge: edge.layer == layer and
                                        edge.prev_node.label == prev_node_label and
                                        edge.current_node.label == node_label,
                           self.edge_lst))[0]

    def search_node(self, label, layer):
        return list(filter(lambda node: node.label == label,
                           self.node_list[layer]))[0]

    def get_z_coef(self):
        if self.z_coef < 0:
            self.z_coef = sum([node.get_alpha_cost()
                                        for node in self.node_list[len(self.X) + 1]])
        return self.z_coef

    def get_marginal_probability(self, prev_y, current_y, layer):
        edge = self.search_edge(prev_y, current_y, layer)
        p = (edge.get_path_cost() / self.get_z_coef()) \
                    * self.search_node(prev_y, layer - 1).get_alpha_cost() \
                    * self.search_node(current_y, layer).get_beta_cost()
        return p
