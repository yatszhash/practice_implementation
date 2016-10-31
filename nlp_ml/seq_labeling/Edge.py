class Edge:
    def __init__(self, prev_node, current_node, layer=None):
        self.prev_node = prev_node
        self.current_node = current_node
        self.layer = layer
        self.path_cost_function = None
        self.path_cost = -1

    def get_path_cost(self):
        if self.path_cost < 0:
            self.path_cost = self.path_cost_function(self.prev_node,
                                                     self.current, self.layer)
        return self.path_cost
