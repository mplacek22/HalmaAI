class Node:
    def __init__(self, move=None, score=None, depth=None):
        self.move = move  # Move leading to this node
        self.score = score  # Evaluation score of this node
        self.depth = depth  # Depth of this node in the decision tree
        self.children = []  # List of child nodes

    def add_child(self, child):
        self.children.append(child)
