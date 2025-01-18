import numpy as np
import jax.numpy as jnp

def jaccard_sim(list1, list2):
    set1, set2 = set(list1), set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0


def topoSort(graph):
    arr = []
    theset = set()
    def DFS(n):
        if n not in theset:
            theset.add(n)
            for i, x in enumerate(graph[n]):
                if x != 0:
                    DFS(i)
            arr.append(n)
    for i in range(num_in):
        DFS(i)
    return np.array(arr)[::-1]

def wt_init():
    wt = np.random.normal(scale=0.01)
    while wt == 0:
        wt = np.random.normal(scale=0.01)
    return wt

def softmax(x):
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum()

