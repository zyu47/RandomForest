import parameters


class Node:
    def __init__(self):
        self.items = []
        self.left_child = None
        self.right_child = None
        self.internal_node = False  # Indicate this node is a splitting node or collecting node
        self.item_cnt_cap = parameters.item_cnt_cap  # The maximum number of items in a node before splitting

    def add(self, sample):
        if self.internal_node:
            pass
        else:
            self.items.append(sample)
            if len(self.items) > self.item_cnt_cap:
                self.split_node()

    def split_node(self):
        pass
