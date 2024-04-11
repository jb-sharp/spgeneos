"""
@author: Giovanni Bocchi
@email: giovanni.bocchi1@unimi.it
@institution: University of Milan

Defines non-standard subgraph families.
"""

import networkx as ntx

# Two triangles
two_triangles = ntx.Graph()
two_triangles.add_nodes_from(range(4))
two_triangles.add_edge(0, 1)
two_triangles.add_edge(1, 2)
two_triangles.add_edge(2, 3)
two_triangles.add_edge(3, 0)
two_triangles.add_edge(2, 0)

triangles = [(two_triangles, 4)]

# Rigid graph 1
rigid1 = ntx.Graph()
rigid1.add_nodes_from(range(7))
rigid1.add_edge(0,1)
rigid1.add_edge(1,2)
rigid1.add_edge(2,3)
rigid1.add_edge(1,4)
rigid1.add_edge(4,5)
rigid1.add_edge(5,6)

# Rigid graph 2
rigid2 = ntx.Graph()
rigid2.add_nodes_from(range(9))
rigid2.add_edge(0,1)
rigid2.add_edge(1,2)
rigid2.add_edge(2,0)
rigid2.add_edge(0,3)
rigid2.add_edge(1,4)
rigid2.add_edge(4,5)
rigid2.add_edge(2,6)
rigid2.add_edge(6,7)
rigid2.add_edge(7,8)

# Rigid graph 3
rigid3 = ntx.Graph()
rigid3.add_nodes_from(range(11))
rigid3.add_edge(0,1)
rigid3.add_edge(1,2)
rigid3.add_edge(2,3)
rigid3.add_edge(1,4)
rigid3.add_edge(4,5)
rigid3.add_edge(5,6)
rigid3.add_edge(1,7)
rigid3.add_edge(7,8)
rigid3.add_edge(8,9)
rigid3.add_edge(9,10)

rigids = [(rigid1, 7), (rigid2, 9)] #, (rigid3, 11)]

# Triangle + 1 edge per node #1
t1 = ntx.cycle_graph(3)
t1.add_edge(0, 3)
# Triangle + 1 edge per node #2
t11 = ntx.cycle_graph(3)
t11.add_edge(0, 3)
t11.add_edge(1, 4)
# Triangle + 1 edge per node #3
t111 = ntx.cycle_graph(3)
t111.add_edge(0, 3)
t111.add_edge(1, 4)
t111.add_edge(2, 5)

# Triangle + 2 edges per node #1
t2 = ntx.cycle_graph(3)
t2.add_edge(0, 3)
t2.add_edge(0, 4)
# Triangle + 2 edges per node #2
t22 = ntx.cycle_graph(3)
t22.add_edge(0, 3)
t22.add_edge(0, 4)
t22.add_edge(1, 5)
t22.add_edge(1, 6)

# Triangle + 2 edges per node #3
t222 = ntx.cycle_graph(3)
t222.add_edge(0, 3)
t222.add_edge(0, 4)
t222.add_edge(1, 5)
t222.add_edge(1, 6)
t222.add_edge(2, 7)
t222.add_edge(2, 8)

# Triangle + 3 edges per node #1
t3 = ntx.cycle_graph(3)
t3.add_edge(0, 3)
t3.add_edge(0, 4)
t3.add_edge(0, 5)

# Square + 1 edge per node #1
s1 = ntx.cycle_graph(4)
s1.add_edge(0, 4)
# Square + 1 edge per node #2
s11 = ntx.cycle_graph(4)
s11.add_edge(0, 4)
s11.add_edge(1, 5)
# Square + 1 edge per node #3
s111 = ntx.cycle_graph(4)
s111.add_edge(0, 4)
s111.add_edge(1, 5)
s111.add_edge(2, 6)
# Square + 1 edge per node #4
s1111 = ntx.cycle_graph(4)
s1111.add_edge(0, 4)
s1111.add_edge(1, 5)
s1111.add_edge(2, 6)
s1111.add_edge(3, 7)

# Square + 2 edges per node #1
s2 = ntx.cycle_graph(4)
s2.add_edge(0, 4)
s2.add_edge(0, 5)

# Square + 3 edges per node #1
s3 = ntx.cycle_graph(4)
s3.add_edge(0, 4)
s3.add_edge(0, 5)
s3.add_edge(0, 6)

# Pentagon + 1 edge per node #1
p1 = ntx.cycle_graph(5)
p1.add_edge(0, 5)
# Pentagon + 1 edge per node #1
p11 = ntx.cycle_graph(5)
p11.add_edge(0, 5)
p11.add_edge(1, 6)
# Pentagon + 1 edge per node #1
p111 = ntx.cycle_graph(5)
p111.add_edge(0, 5)
p111.add_edge(1, 6)
p111.add_edge(2, 7)
# Pentagon + 1 edge per node #1
p1111 = ntx.cycle_graph(5)
p1111.add_edge(0, 5)
p1111.add_edge(1, 6)
p1111.add_edge(2, 7)
p1111.add_edge(3, 8)
# Pentagon + 1 edge per node #1
p11111 = ntx.cycle_graph(5)
p11111.add_edge(0, 5)
p11111.add_edge(1, 6)
p11111.add_edge(2, 7)
p11111.add_edge(3, 8)
p11111.add_edge(4, 9)

# Pentagon + 2 edges per node #1
p2 = ntx.cycle_graph(5)
p2.add_edge(0, 5)
p2.add_edge(0, 6)

# Pentagon + 3 edges per node #1
p3 = ntx.cycle_graph(5)
p3.add_edge(0, 5)
p3.add_edge(0, 6)
p3.add_edge(0, 7)

# Exagon + 1 edge per node #1
e1 = ntx.cycle_graph(6)
e1.add_edge(0, 6)
# Exagon + 1 edge per node #2
e11 = ntx.cycle_graph(6)
e11.add_edge(0, 6)
e11.add_edge(1, 7)
# Exagon + 1 edge per node #3
e111 = ntx.cycle_graph(6)
e111.add_edge(0, 6)
e111.add_edge(1, 7)
e111.add_edge(2, 8)
# Exagon + 1 edge per node #4
e1111 = ntx.cycle_graph(6)
e1111.add_edge(0, 6)
e1111.add_edge(1, 7)
e1111.add_edge(2, 8)
e1111.add_edge(3, 9)
# Exagon + 1 edge per node #5
e11111 = ntx.cycle_graph(6)
e11111.add_edge(0, 6)
e11111.add_edge(1, 7)
e11111.add_edge(2, 8)
e11111.add_edge(3, 9)
e11111.add_edge(4, 10)
# Exagon + 1 edge per node #6
e111111 = ntx.cycle_graph(6)
e111111.add_edge(0, 6)
e111111.add_edge(1, 7)
e111111.add_edge(2, 8)
e111111.add_edge(3, 9)
e111111.add_edge(4, 10)
e111111.add_edge(5, 11)

# Exagon + 2 edges per node #1
e2 = ntx.cycle_graph(6)
e2.add_edge(0, 6)
e2.add_edge(0, 7)

# Exagon + 3 edges per node #1
e3 = ntx.cycle_graph(6)
e3.add_edge(0, 6)
e3.add_edge(0, 7)
e3.add_edge(0, 8)

# List of tuples of (graph, number of nodes)
augmented1 = [(t1, 4), (t11, 5), (t111, 6),
              (s1, 5), (s11, 6), (s111, 7), (s1111, 8),
              (p1, 6), (p11, 7), (p111, 8), (p1111, 9), (p11111, 10),
              (e1, 7), (e11, 8), (e111, 9), (e1111, 10), (e11111, 11), (e111111, 12)]

augmented2 = [(t2, 5), (t22, 7), (t222, 9), (s2, 6), (p2, 7), (e2, 8)]

augmented3 = [(t3, 6), (s3, 7), (p3, 8), (e3, 9)]
