import sys
import networkx as nx
import pandas as pd
import numpy as np
from scipy.special import loggamma 

def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))

def process(graph: nx.Graph, data: pd.DataFrame):
    values = {} # node -> {values}
    parents = {} # node -> [parents]
    for node in graph.nodes():
        values[node] = sorted(data[node].unique().tolist())
        parents[node] = []
    for edge in graph.edges():
        parent, child = edge
        parents[child] += [parent]
    return values, parents

def get_ijk_value(index: str, row: pd.Series, values: {}, parents: {}, nodes_list: []):
    curr_var = nodes_list[index]
    x_parents = parents[curr_var].copy()
    x_parents_vals = [row.iloc[nodes_list.index(parent)] for parent in x_parents]
    x_parents_dims = [len(values[parent]) for parent in x_parents]
    j = (np.ravel_multi_index(x_parents_vals, x_parents_dims) if len(x_parents) > 0 else 0)
    k = values[curr_var].index(row.iloc[index])
    return index, j, k

def bayesian_score(graph: nx.Graph, data: pd.DataFrame):
    values, parents = process(graph, data)
    nodes = list(graph.nodes)

    # * instantiate empty counts : list of 2d arrays
    counts = [None] * len(nodes)
    for index in range(len(counts)):
        curr_var = nodes[index]
        q = sum([len(values[parent]) for parent in parents[curr_var]])
        q = q if q > 0 else 1
        counts[index] = [None] * q
        for i in range(len(counts[index])):
            counts[index][i] = [0] * len(values[curr_var])

    # * iterate through data to populate the counts
    for row in data.iterrows():
        # print("row " + str(row[0]) + "\n" + str(row[1]))
        data_tuple = row[1]
        for index in range(len(data_tuple)): # 0 -> n - 1
            i, j, k = get_ijk_value(index, data_tuple, values, parents, nodes)
            counts[i][j][k] += 1
            # print("index", index, "i, j, k: ", i, j, k)

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
    return p

def compute(infile, outfile):
    # * process csv into graph
    # data = pd.read_csv("data/large.csv", delimiter=",")
    # var_names = data.columns
    # graph = nx.Graph()
    # graph.add_nodes_from(var_names)

    # dummy test data A -> B <- C (example from video)
    graph = nx.Graph()
    graph.add_nodes_from(['A', 'B', 'C'])
    graph.add_edges_from([('A', 'C'), ('B', 'C')])
    data = pd.read_csv("data/test_data.csv", delimiter=",")
    print(data)

    for row in data.iterrows():
        print(row)
        print("...")

    print("nodes: " + str(graph.nodes()))
    print("edges: " + str(graph.edges()))

    # * calculate bayesian score
    score = bayesian_score(graph, data)
    print("score: " + str(score))

    # print("score: " + str(bayesian_score(test_graph, data)))

    pass


def main():
    # if len(sys.argv) != 3:
    #     raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    # inputfilename = sys.argv[1]
    # outputfilename = sys.argv[2]
    # compute(inputfilename, outputfilename)
    compute(None, None)


if __name__ == '__main__':
    main()
