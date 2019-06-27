
# a node in a graph that has next and prev direction
class Node:
    def __init__(self, name = 'Node'):
        self.next = []
        self.prev = []
        self.name = name

    def copy(self):
        return Node(name = self.name)
        
    def to_str(self):
        return '\t'.join([self.name,
                          ','.join([n.name for n in self.next]), 
                          ','.join([n.name for n in self.prev]),
                          str(id(self))
        ])

# depth first search over a graph with a specified direction
def dfs_iter_helper(node, visited, attr_next = 'next'):
    for i in getattr(node, attr_next):
        for j in dfs_iter_helper(i, visited, attr_next):
            yield j
    if node not in visited:
        yield node
        visited.add(node)
def dfs_iter(node, attr_next = 'next'):
    visited = set()
    for i in dfs_iter_helper(node, visited, attr_next):
        yield i
# breath first search over a graph with a specified direction
def bfs_iter(node, attr_next = 'next'):
    q = [node]
    cur_idx = 0
    while True:
        if cur_idx >= len(q):
            break
        cur = q[cur_idx]
        for i in getattr(cur, attr_next):
            if i not in q:
                q.append(i)
        yield cur
        cur_idx += 1

# fill in prev info according to next, do it in place
def double_connect(root):
    for n in bfs_iter(root):
        n.prev = []
    for n in bfs_iter(root):
        for i in n.next:
            i.prev.append(n)
    return root

# return a new graph which is a subgraph starting from node along
# a speficied direction
def subgraph(node, direction = 'next'):
    new_root = node.copy()
    copied = {node : new_root}
    for cur_node in bfs_iter(node, direction):
        cur_node_copy = copied[cur_node]
        for next_node in getattr(cur_node, direction):
            if next_node in copied:
                cur_node_copy.next.append(copied[next_node])
            else:
                new_copy_node = next_node.copy()
                cur_node_copy.next.append(new_copy_node)
                copied[next_node] = new_copy_node
    return double_connect(new_root)


def print_graph(root):
    for i in bfs_iter(root):
        print(i.to_str())


# dfs_prop_iter aims to iterate over a graph rooted at 'node'
# with a constrain that current node be visited only if all the previous
# nodes have been visited
# this is for backprobagation iteration
def all_visited(nlist, visited):
    for n in nlist:
        if n not in visited:
            return False
    return True
def bfs_prop_iter_helper(node, visited, attr_next = 'next', attr_prev = 'prev'):
    if all_visited(getattr(node, attr_prev), visited):
        if node not in visited:
            yield node
            visited.add(node)
        for i in getattr(node, attr_next):
            for j in bfs_prop_iter_helper(i, visited, attr_next, attr_prev):
                yield j
def bfs_prop_iter(root, attr_next = 'next', attr_prev = 'prev'):
    visited = set()
    for n in bfs_prop_iter_helper(root, visited, attr_next, attr_prev):
        yield n


        
if __name__ == '__main__':
    pass
