class Node:
    def __init__(self, word, label, layer):
        self.label = label
        self.prev_nodes = {}
        self.next_nodes = {}
        self.alpha_cost = -1
        self.best_cost = -1
        self.word = word
        self.layer = layer

    def get_alpha_cost(self):
        if self.alpha_cost < 0 and self.label == "BOS":
            self.alpha_cost = 1

        elif self.alpha_cost < 0:
            self.alpha_cost = sum([prev_node.get_alpha_cost() * edge.path_cost
                                   for prev_node, edge in self.prev_nodes.items()])

        return self.alpha_cost

    def get_beta_cost(self):
        if self.best_cost < 0 and self.label == "EOS":
            self.best_cost = 1
        elif self.best_cost < 0:
            self.best_cost = sum([next_node.get_beta_cost() * edge.path_cost
                                  for next_node, edge in self.next_nodes.items()])
        return self.best_cost

    def add_prev_nodes(self, prev_node, edge):
        self.prev_nodes[prev_node] = edge

    def add_next_nodes(self, next_node, edge):
        self.next_nodes[next_node] = edge
