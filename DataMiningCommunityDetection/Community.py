class Community:
    def __init__(self, nodes):
        self.nodes = nodes

    def __str__(self):
        return str(self.nodes)

    def __hash__(self):
        return hash(str(self.nodes))

    def __eq__(self,other):
        return hash(self) == hash(other)


    def merge_community(self,other):
        return Community(self.nodes[:] + other.nodes[:])