import sys
import networkx as nx
import pandas as pd
import numpy as np
import random
import time
import matplotlib.pyplot as plt

from scipy.special import loggamma 
from networkx.drawing.nx_agraph import write_dot

def write_gph(dag, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(edge[0], edge[1]))

def process(graph: nx.Graph, data: pd.DataFrame):
    num_values = {} # node -> # values it can take, aka r
    parents = {} # node -> [parents]
    for node in graph.nodes():
        num_values[node] = max(data[node])
        parents[node] = []
    for edge in graph.edges():
        parent, child = edge
        parents[child] += [parent]
    return num_values, parents

def get_ijk_value(index: str, row: pd.Series, num_values: {}, parents: {}, nodes_list: []):
    curr_var = nodes_list[index]
    # print(curr_var)
    x_parents = parents[curr_var].copy()
    x_parents_indices = [row.iloc[nodes_list.index(parent)] - 1 for parent in x_parents]
    x_parents_dims = [num_values[parent] for parent in x_parents]
    # print("parents", x_parents)
    # print("parent dims", x_parents_dims)
    # print("parent inds", x_parents_indices)
    j = (np.ravel_multi_index(x_parents_indices, x_parents_dims) if len(x_parents) > 0 else 0)
    k = row.iloc[index] - 1
    return index, j, k

def bayesian_score(graph: nx.Graph, data: pd.DataFrame):
    num_values, parents = process(graph, data)
    nodes = list(graph.nodes)

    # * instantiate empty counts : list of 2d arrays
    counts = [None] * len(nodes)
    for index in range(len(counts)):
        curr_var = nodes[index]
        q = 1
        for parent in parents[curr_var]:
            q *= num_values[parent]
        counts[index] = [None] * q
        for i in range(len(counts[index])):
            counts[index][i] = [0] * num_values[curr_var]

    # * iterate through data to populate the counts
    for row in data.iterrows():
        # print(row)
        data_tuple = row[1]
        for index in range(len(data_tuple)): # 0 -> n - 1
            i, j, k = get_ijk_value(index, data_tuple, num_values, parents, nodes)
            # print(i, j, k)
            # print(counts)
            counts[i][j][k] += 1

    # print(counts)
    
    # * calculate bayesian score
    p = 0
    for i in range(len(nodes)):
        for j in range(len(counts[i])):
            row = counts[i][j]
            p += loggamma(len(row))
            p -= loggamma(len(row) + sum(row))
            for k in range(len(row)):
                p += loggamma(1 + row[k])
                p -= loggamma(1)
    return p, counts

def bayesian_score_from_prev_graph(graph: nx.Graph, child: str, data: pd.DataFrame, prev_p: int, prev_counts: [[[]]]):
    num_values, parents = process(graph, data)
    nodes = list(graph.nodes)
    child_index = nodes.index(child)
    counts = prev_counts.copy()

    # * "undo" effect of previous child counts on p
    p = prev_p
    for j in range(len(counts[child_index])):
        row = counts[child_index][j]
        p -= loggamma(len(row))
        p += loggamma(len(row) + sum(row))
        for k in range(len(row)):
            p -= loggamma(1 + row[k])
            p += loggamma(1)

    # * reset child's 2d counts array
    q = 1
    for parent in parents[child]:
        q *= num_values[parent]
    counts[child_index] = [None] * q
    for i in range(q):
        counts[child_index][i] = [0] * num_values[child]

    # * iterate through data to populate the counts
    for row in data.iterrows():
        # print(row)
        data_tuple = row[1]
        i, j, k = get_ijk_value(child_index, data_tuple, num_values, parents, nodes)
        counts[i][j][k] += 1

    # print(counts)
    
    # * update p with new child counts
    for j in range(len(counts[child_index])):
        row = counts[child_index][j]
        p += loggamma(len(row))
        p -= loggamma(len(row) + sum(row))
        for k in range(len(row)):
            p += loggamma(1 + row[k])
            p -= loggamma(1)
    return p, counts

def find_graph_given_data(graph: nx.Graph, data: pd.DataFrame, max_parents=2):
    ordering = list(graph.nodes)
    random.shuffle(ordering)
    best_score = None
    score, counts = bayesian_score(graph, data)
    for i_index, i in enumerate(ordering[1:-1]):
        if i_index % 10 == 0:
            print("intermediate graph:", str(graph.edges))
            print("score", str(best_score))
        print("finding parents for node", str(i_index), str(i))
        best_score = score
        best_parent = None
        best_counts = None
        found_all_parents = False
        num_parents_found = 0
        while not found_all_parents and (max_parents is None or num_parents_found < max_parents):
            print("finding parent #", str(num_parents_found + 1))
            for j in ordering[0:i_index]:
                print("testing node", str(j), "as parent")
                if (j, i) not in graph.edges:
                    graph.add_edge(j, i)
                    # print("add edge (" + str(j) + ", " + str(i) + ")")
                    temp_score, new_counts = bayesian_score_from_prev_graph(graph, i, data, score, counts)
                    if temp_score > best_score:
                        best_score, best_parent, best_counts = temp_score, j, new_counts
                    graph.remove_edge(j, i)
            if best_score > score:
                score = best_score
                num_parents_found += 1
                graph.add_edge(best_parent, i)
                counts = best_counts
            else:
                found_all_parents = True
    return graph, best_score

def compute(infile, outfile, restarts=1, max_parents=2):
    start_time = time.time()
    times = [] # ("name", time)

    # * process csv into graph
    data = pd.read_csv(infile, delimiter=",")
    var_names = data.columns
    graph = nx.DiGraph()
    graph.add_nodes_from(var_names)
    # graph.add_edges_from([('parent1', 'child1'), ('parent2', 'child2'), ('parent3', 'child3')])
    # graph.add_edges_from([('portembarked', 'passengerclass'), ('fare', 'passengerclass'), ('fare', 'sex'), ('numparentschildren', 'sex'), ('numparentschildren', 'numsiblings'), ('passengerclass', 'survived'), ('sex', 'survived')])

    # dummy test data A -> B <- C (example from video)
    # graph = nx.DiGraph()
    # graph.add_nodes_from(['A', 'B', 'C'])
    # graph.add_edges_from([('A', 'C'), ('B', 'C')])
    # data = pd.read_csv("data/test_data.csv", delimiter=",")
    
    # print(data)
    print("nodes: " + str(graph.nodes()))
    print("edges: " + str(graph.edges()))

    # * calculate bayesian score
    score, counts = bayesian_score(graph, data)
    print("starting score: " + str(score))

    # * find best graph representation
    best_graph, best_score = graph, None
    best_trial = None
    for i in range(restarts):
        trial_start_time = time.time()
        print("running attempt: " + str(i))
        new_graph, new_score = find_graph_given_data(graph.copy(), data, max_parents=max_parents)
        print("new graph:", new_graph.edges)
        print("new score: " + str(new_score))
        if best_score is None or new_score > best_score:
            best_score = new_score
            best_graph = new_graph
            best_trial = i
        times += [("trial " + str(i), round(time.time() - trial_start_time, 3))]
    times += [("total", round(time.time() - start_time, 3))]

    write_gph(best_graph, outfile)
    print(best_graph.edges)
    print(best_score)
    print("best graph time", times[best_trial], "total", times[-1])
    nx.draw_networkx(best_graph, arrows=True)
    plt.show()
    pass


def main():
    # if len(sys.argv) != 3:
    #     raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    # inputfilename = sys.argv[1]
    # outputfilename = sys.argv[2]
    # compute(inputfilename, outputfilename)
    
    # compute("data/medium.csv", "medium.gph", restarts=5, max_parents=None)
    compute("data/large.csv", "large.gph", restarts=1, max_parents=2)
    # compute("example/example.csv", "test.gph", restarts=1, max_parents=2)


if __name__ == '__main__':
    main()
