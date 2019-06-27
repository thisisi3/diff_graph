from dg.graph import *



if __name__ == '__main__':
    n1 = Node(name = '1')
    n2 = Node(name = '2')
    n3 = Node(name = '3')
    n4 = Node(name = '4')
    n5 = Node(name = '5')
    n6 = Node(name = '6')
    n7 = Node(name = '7')
    n8 = Node(name = '8')
    
    n1.next = [n2, n3]
    n2.next = [n4, n6]
    n3.next = [n4, n5]
    n4.next = [n6, n5]
    n5.next = []
    n6.next = [n7]
    n7.next = []
    n8.next = [n3]
    print('-' * 100)
    print('Graph before double connect:')
    print_graph(n1)
    print('Graph after double connect:')
    print_graph(double_connect(n1))
    n3.prev.append(n8)
    
    print('-' * 100)
    print('bfs test:')
    print('start from node 1')
    for n in bfs_iter(n1):
        print(n.name)
    print('start from node 4')
    for n in bfs_iter(n4):
        print(n.name)
            
    print('start from node 4, reversely')
    for n in bfs_iter(n4, 'prev'):
        print(n.name)
        
    print('-' * 100)
    print('dfs test:')
    print('start from node 1')
    for n in dfs_iter(n1):
        print(n.name)
    print('start from node 4')
    for n in dfs_iter(n4):
        print(n.name)
    print('start from node 4, reversely')
    for n in dfs_iter(n4, 'prev'):
        print(n.name)
        
    # test subgraph
    print('-' * 100)
    print('Subgraph from node 4, forward:')
    new_root = subgraph(n4, 'next')
    print_graph(new_root)
    
    # test utils for backward() and forward()
    print('-' * 100)
    print('Subgraph from node 6')
    new_root = subgraph(n6, 'prev')
    print_graph(new_root)

    print('forward() via dfs')
    for n in dfs_iter(new_root):
        print(n.name)
    
    print('backward() via dfs_prop_iter')
    for n in bfs_prop_iter(new_root):
        print(n.name)
    
