
# disjoint-set forests using union-by-rank and path compression (sort of).


class UniElt:

    def __init__(self, rank, size, p):
        self.rank = rank
        self.size = size
        self.p = p

    def get_size(self):
        return self.size;


class Universe:

    def __init__(self, num_elements):
        self.elts = []
        self.num = num_elements
        for i in range(0, num_elements):
            self.elts.append(UniElt(0, 1, i))

    def find(self, x):
        # finds which component the vertex belongs to
        y = x
        while y != self.elts[y].p:
            y = self.elts[y].p

        self.elts[x].p = y
        return y

    def join(self, x, y):
        # Join x,y based on rank
        if self.elts[x].rank > self.elts[y].rank:
            self.elts[y].p = x
            self.elts[x].size += self.elts[y].size
        else:
            self.elts[x].p = y
            self.elts[y].size += self.elts[x].size

            if self.elts[x].rank == self.elts[y].rank:
                self.elts[y].rank += 1

        self.num -= 1

    def get_size(self, x):
        e = self.elts[x]
        return e.get_size()

    def num_sets(self):
        return self.num
