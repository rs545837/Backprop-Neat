import jax
import jax.numpy as jnp
from jax import vmap, grad, jit, lax
from utils import *
from problems import *

import numpy as np
import json
import copy
import random
import pickle 
import math
import os
import time
import shutil

import networkx as nx
import matplotlib.pyplot as plt

with open('configBP.json', 'r') as file:
    config = json.load(file)

num_in = config['num_in']
num_out = config['num_out']
init_pop = config['init_pop']
weight_range = config['weight_range']
edge_tries = config['edge_tries']
num_generations = config['num_generations']
keep_percent = config['keep_percent']

c1 = config['excess_const']
c2 = config['disjoint_const']
# c3 = config['weight_const']
c4 = config['jaccard_const'] #
N = config['genome_size_norm']
compatibility_threshold = config['compatibility threshold']

wt_mutation = config['wt_mutation']
single_wt = config['single_wt']
nudge_range = config['nudge_range']
new_link_chance = config['new_link_chance']
new_node_chance = config['new_node_chance']
stagnation_cutoff = config['stagnation_cutoff']
temperature = config["temperature"]
num_elites = config["num_elites"]
clamp = config["clamp"]

num_backprop_steps = config["num_backprop_steps"]
learning_rate = config["learning_rate"]
connection_penalty = config["connection_penalty"]

MAX_NODE_CT = config["max_node_count"]

def neat_act(x):
    return 1 / (1 + jnp.exp(-4.9*x))

def jnp_relu(x):
    return jnp.maximum(0, x)

def identity(x):
    return x # LOL

fun_enum = [identity, jnp.absolute, jnp.square, jnp.sin, jnp_relu, neat_act]

def getActivations(genelist):
    activation_list = np.zeros(MAX_NODE_CT, dtype=int)
    for gene in genelist:
        if gene.enable:
            activation_list[gene.out_node] = gene.activation
    return activation_list


def visualize_graph(adj_mat, gen_num, fitness, activations, is_best=False):
    global MAX_NODE_CT, num_in, num_out

    # Create a directed graph
    G = nx.DiGraph()

    # Add edges to the graph
    for i in range(MAX_NODE_CT):
        for j in range(MAX_NODE_CT):
            if adj_mat[i][j] != 0:
                G.add_edge(i, j, weight=adj_mat[i][j])

    # Custom positions for the nodes
    pos = {}
    # Place nodes 0-11 at the bottom in a row
    for i in range(num_in):
        pos[i] = (i, 0)
    # Place nodes 12-14 in a row at the top
    for i in range(num_in, num_in + num_out):
        pos[i] = (i - 6, 2)
    # Place any additional nodes in the middle
    middle_nodes = [n for n in G.nodes() if n not in pos]
    middle_x = 6  # Starting x position for middle nodes
    for i, node in enumerate(middle_nodes):
        pos[node] = (middle_x + i, 1)

    # Edge weights
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    # Normalize weights for color mapping
    weights_normalized = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))

    # Create a figure and axes with white background
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Define activation function labels
    activation_labels = {0: 'eye', 1: 'abs', 2: 'square', 3: 'sin', 4: 'relu', 5: 'sig'}

    # Create a color map for activations
    activation_colors = plt.colormaps['Set1']
    node_colors = [activation_colors(activations[i]) for i in G.nodes()]

    # Draw the graph with original sizes
    nx.draw(G, pos, ax=ax, 
            edge_color=weights_normalized,
            edge_cmap=plt.cm.viridis,
            node_color=node_colors,
            with_labels=True,
            node_size=500,
            font_size=10,
            font_color='black',
            width=1.0)

    # Add activation function labels with original size
    for node in G.nodes():
        x, y = pos[node]
        label = activation_labels.get(activations[node], '')
        ax.text(x, y + 0.1, label, 
                fontsize=8,
                ha='center',
                color='black',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    # Create a color bar with original size
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                              norm=plt.Normalize(vmin=min(weights), vmax=max(weights)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Edge Weights', color='black', fontsize=10)
    cbar.ax.yaxis.set_tick_params(color='black')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')

    # Add annotations with original size and positioning
    title_text = f"Generation: {gen_num}\nFitness: {fitness:.6f}"
    if is_best:
        title_text = "BEST NETWORK\n" + title_text
    
    plt.text(0.1, 0.98, title_text,
             transform=ax.transAxes,
             fontsize=18,
             color='black',
             verticalalignment='top',
             horizontalalignment='left',
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    # Ensure the output folder exists
    output_folder = "bpneat_graphs"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the plot with appropriate filename
    filename = f"best_{gen_num}.png" if is_best else f"{gen_num}.png"
    plt.savefig(os.path.join(output_folder, filename),
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.close()


class Gene:
    def __init__(self, in_node, out_node, enable, innov_num, activation=0):
        self.in_node = in_node
        self.out_node = out_node
        self.enable = enable
        self.innov_num = innov_num
        self.activation = activation


def geneGraph(genelist):
    graph = np.zeros((MAX_NODE_CT, MAX_NODE_CT))
    for gene in genelist:
        if gene.enable:
            i, o = gene.in_node, gene.out_node
            graph[i][o] = wt_init()
    return graph, np.where(graph == 0, 0, 1)

class Genome:

    innov_num = num_in * num_out
    mutations = dict()
    max_nodes = num_in + num_out

    def __init__(self, *args):
        self.fitness = None

        if args:
            genelist = args[0]
            self.genelist = genelist
            self.node_ct = 1 + max(max(gene.in_node, gene.out_node) for gene in genelist)

        else:
            self.node_ct = num_in + num_out
            self.genelist = []
            for i in range(num_in):
                for j in range(num_in, self.node_ct):
                    innov = i + j * num_in
                    self.genelist.append(Gene(i, j, True, innov, activation=5))

        self.connection_ct = len([gene for gene in self.genelist if gene.enable])
        self.graph, self.bools = geneGraph(self.genelist)
        self.toposort = topoSort(self.graph)
        self.activations = getActivations(self.genelist)
        
    def mutate_new_node(self):
        if self.node_ct < MAX_NODE_CT:
            idx = np.random.choice([i for i, x in enumerate(self.genelist) if x.enable])
            self.genelist[idx].enable = False
            i, o = self.genelist[idx].in_node, self.genelist[idx].out_node
            ct, act = self.node_ct, self.genelist[idx].activation
            new_act = random.randint(1, len(fun_enum) - 1)
            if ('node', i, o, new_act) in Genome.mutations:
                innov = Genome.mutations[('node', i, o, new_act)]
            else:
                innov = Genome.innov_num
                Genome.mutations[('node', i, o, new_act)] = innov
                Genome.innov_num += 2

            self.genelist.append(Gene(i, ct, True, innov, activation=act))
            self.genelist.append(Gene(ct, o, True, innov, activation=new_act))
            self.node_ct += 1
            self.connection_ct += 1
            Genome.max_nodes = max(Genome.max_nodes, self.node_ct)

            self.bools[i][o], self.bools[i][ct], self.bools[ct][o] = 0, 1, 1
            self.graph[i][o], self.graph[i][ct], self.graph[ct][o] = 0, wt_init(), wt_init()
            self.activations = getActivations(self.genelist)

    def mutate_new_edge(self):
        for _ in range(edge_tries):
            ins = [idx for idx, x in enumerate(self.toposort) if x not in range(num_in, num_in + num_out)]
            if ins: 
                i = np.random.choice(ins)
            else:
                break
            outs = [idx for idx, x in enumerate(self.toposort) if idx > i and x not in range(num_in)]
            if outs:
                o = np.random.choice(outs)
            else:
                break

            if self.bools[i][o]:
                continue
            else:
                if ('edge', i, o) in Genome.mutations:
                    innov = Genome.mutations[('edge', i, o)]
                else:
                    innov = Genome.innov_num
                    Genome.mutations[('edge', i, o)] = innov
                    Genome.innov_num += 1

                self.genelist.append(Gene(i, o, True, innov, activation=self.activations[o]))
                self.graph[i][o] = wt_init()
                self.bools[i][o] = 1
                self.connection_ct += 1
                self.activations = getActivations(self.genelist)
                break


def crossover(genome1, genome2):
    if genome2.fitness >= genome1.fitness:
        genome1, genome2 = genome2, genome1

    genelist1, genelist2 = genome1.genelist, genome2.genelist
    newgenes = []
    pt1, pt2 = 0, 0

    while pt1 < len(genelist1) and pt2 < len(genelist2):
        gene1, gene2 = genelist1[pt1], genelist2[pt2]

        if gene1.innov_num == gene2.innov_num:
            newgenes.append(copy.copy(gene1)) if np.random.randint(0, 2) else newgenes.append(copy.copy(gene2))
            pt1 += 1
            pt2 += 1
        elif gene1.innov_num < gene2.innov_num:
            newgenes.append(copy.copy(gene1))
            pt1 += 1
        else:
            pt2 += 1

    newgenes.extend(copy.copy(gene) for gene in genelist1[pt1:])
    return Genome(newgenes)

def num_matches(g1, g2):
    pt1, pt2, ct, summed = 0, 0, 0, 0
    while pt1 < len(g1) and pt2 < len(g2):
        if g1[pt1].innov_num == g2[pt2].innov_num:
            ct += 1
            pt1 += 1
            pt2 += 1
        elif g1[pt1].innov_num < g2[pt2].innov_num:
            pt1 += 1
        else:
            pt2 += 1
    return ct

def compatibility(genome1, genome2):
    g1, g2 = genome1.genelist, genome2.genelist
    excess = max(len(g1), len(g2)) - min(len(g1), len(g2))
    matches = num_matches(g1, g2)
    disjoint = min(len(g1), len(g2)) - matches
    jacc = jaccard_sim(genome1.activations, genome2.activations)
    compatibility = ((c1 * excess) / N) + ((c2 * disjoint) / N) + (jacc * c4) # will need to adjust c4, maybe have it multiply expr
    return compatibility

def sharing_func(genome1, genome2):
    return True if compatibility(genome1, genome2) < compatibility_threshold else False


class Species:
    def __init__(self, *args):
        if len(args) == 1:
            self.members = [args[0]]
            self.rep = copy.deepcopy(args[0])
        elif len(args) == 2:
            self.members = args[0]
            self.rep = args[1]
        self.fitness = 0
        self.stagnation_num = 0
        self.max_fitness = None

    def assign_rep(self):
        self.rep = copy.deepcopy(self.members[0])
        
    def is_member(self, genome):
        if sharing_func(self.rep, genome):
            self.members.append(genome)
            return True
        return False

    def new_gen_spec(self, num_members):
        self.members = sorted(self.members, key=lambda x: x.fitness, reverse=True)

        if len(self.members) > 2:
            num_to_keep = int(len(self.members) * keep_percent)
            self.members = self.members[:num_to_keep]
        
        if len(self.members) < num_members:
            if len(self.members) == 1:
                self.members.append(copy.deepcopy(self.members[0]))

            probs = softmax(np.array([x.fitness for x in self.members]))
            num_crosses = num_members - len(self.members)

            for i in range(num_crosses):
                org1, org2 = np.random.choice(self.members[:len(probs)], 2, p=probs, replace=True)
                self.members.append(crossover(org1, org2))

            for org in self.members:
                org.graph, org.bools = geneGraph(org.genelist)
                org.toposort = topoSort(org.graph)
                org.activations = getActivations(org.genelist)

            for i in range(num_elites, len(self.members)):
                org = self.members[i]

                prob = np.random.uniform()
                if prob < new_node_chance:
                    org.mutate_new_node()
                    org.toposort = topoSort(org.graph)

                prob = np.random.uniform()
                if prob < new_link_chance:
                    org.mutate_new_edge()

                org.activations = getActivations(org.genelist)

    def update_fitness(self):
        self.fitness = 0
        for org in self.members:
            self.fitness += org.fitness
        if self.max_fitness and self.fitness <= self.max_fitness:
            self.stagnation_num += 1
        else:
            self.stagnation_num = 0
        self.max_fitness = self.fitness if not self.max_fitness else max(self.fitness, self.max_fitness)

def population_stack(orglist):
    the_graphs = np.stack([org.graph for org in orglist])
    the_bools = np.stack([org.bools for org in orglist])
    the_activations = np.stack([org.activations for org in orglist])

    # use each organism's toposort to do the reindexing on graphs, bools, activations, etc.
    for i in range(len(orglist)):
        indices = np.arange(MAX_NODE_CT)
        indices[:len(orglist[i].toposort)] = orglist[i].toposort
        the_activations[i] = the_activations[i][indices]
        the_graphs[i] = the_graphs[i][indices]
        the_graphs[i] = np.array([row[indices] for row in the_graphs[i]])
        the_bools[i] = the_bools[i][indices]
        the_bools[i] = np.array([row[indices] for row in the_bools[i]])

    return the_graphs, the_bools, the_activations

def output(graph, bools, activation, x):
    resarray = jnp.zeros(MAX_NODE_CT)
    resarray = resarray.at[:num_in].set(x)
    for i in range(Genome.max_nodes):
        act_val = lax.switch(activation[i], fun_enum, resarray[i])
        resarray = resarray.at[i].set(act_val)
        for j in range(Genome.max_nodes):
            tmp = bools[i][j] * resarray[i] * graph[i][j]
            resarray = resarray.at[j].add(tmp)
    return resarray[num_in:num_in+num_out]


def loss(vmap_output_fn, graph, bools, activation, x, y):
    preds = vmap_output_fn(graph, bools, activation, x)
    return -jnp.mean(y * (jnp.log(preds + 1e-6)))

def eval_org(graph, bools, activation, x, y):
    vmap_output_fn = jit(vmap(output, in_axes=(None, None, None, 0)))
    grad_fn = grad(loss, argnums=1)
    
    # Initialize adaptive learning rate parameters
    initial_lr = learning_rate
    min_lr = 1e-6
    max_lr = 0.1
    lr_reduction_factor = 0.5
    
    # Use JAX arrays for loss tracking
    best_loss = jnp.array(float('inf'))
    current_lr = jnp.array(initial_lr)
    
    def step_fn(step, carry):
        graph, best_loss, current_lr = carry
        
        # Calculate current loss and gradients
        current_loss = loss(vmap_output_fn, graph, bools, activation, x, y)
        grad_graph = grad_fn(vmap_output_fn, graph, bools, activation, x, y)
        
        # Update learning rate using JAX's where
        best_loss = jnp.minimum(best_loss, current_loss)
        current_lr = jnp.where(
            current_loss > best_loss,
            jnp.maximum(min_lr, current_lr * lr_reduction_factor),
            current_lr
        )
        
        # Update weights
        graph = graph - current_lr * grad_graph
        
        return (graph, best_loss, current_lr)
    
    # Run training loop using lax.fori_loop
    init_carry = (graph, best_loss, current_lr)
    final_graph, final_best_loss, _ = lax.fori_loop(
        0, num_backprop_steps, 
        lambda i, carry: step_fn(i, carry), 
        init_carry
    )
    
    total_loss = loss(vmap_output_fn, final_graph, bools, activation, x, y)
    average_loss = jnp.mean(total_loss)
    return average_loss, final_graph

eval_orgs = vmap(eval_org, in_axes=(0, 0, 0, None, None))

# ty chatgpt for adding the matplotlib stuff <3
def test_viz(graph, bools, acts, x, y, gen_num, is_best=False):
    v_output = vmap(output, in_axes=(None, None, None, 0))

    # Split the data into training and testing sets
    train_x, train_y = jnp.array(x[:len(x)//2]), jnp.array(y[:len(y)//2])
    test_x, test_y = jnp.array(x[len(x)//2:]), jnp.array(y[len(y)//2:])

    # Generate predictions for training and testing sets
    train_preds = np.round(v_output(graph, bools, acts, train_x)).reshape(-1).astype(int)
    test_preds = np.round(v_output(graph, bools, acts, test_x)).reshape(-1).astype(int)

    # Calculate accuracy
    train_acc = np.sum(train_preds != train_y) / len(train_y)
    test_acc = np.sum(test_preds != test_y) / len(test_y)

    print(f"Train accuracy: {train_acc} \nTest accuracy: {test_acc}")

    def plot_data(x, preds, y, dataset_type, gen_num, accuracy):
        plt.figure(figsize=(12,9))
        
        # Define a better color scheme
        class_0_fill = '#4b6bfb'    # A richer blue for class 0
        class_1_fill = '#FFB86C'    # A warm orange for class 1
        correct_edge = '#50FA7B'    # A vibrant green for correct predictions
        incorrect_edge = '#FF5555'  # A bright red for incorrect predictions
        
        # Set style - using a compatible style
        plt.style.use('dark_background')
        
        # Set custom colors and styles
        plt.rcParams['figure.facecolor'] = '#282a36'  # Dark background
        plt.rcParams['axes.facecolor'] = '#282a36'
        plt.rcParams['text.color'] = '#f8f8f2'        # Light text
        plt.rcParams['axes.labelcolor'] = '#f8f8f2'
        plt.rcParams['xtick.color'] = '#f8f8f2'
        plt.rcParams['ytick.color'] = '#f8f8f2'
        
        # Plot each point
        for i in range(len(x)):
            # Determine colors and markers
            fill_color = class_1_fill if y[i] == 1 else class_0_fill
            edge_color = incorrect_edge if preds[i] == y[i] else correct_edge  # Reversed the logic
            
            plt.scatter(x[i, 0], x[i, 1], 
                       c=fill_color,
                       edgecolor=edge_color,
                       linewidth=0.75,
                       s=40,
                       alpha=0.8,
                       label=f'Class {y[i]} ({"Correct" if preds[i] == y[i] else "Incorrect"})' if i == 0 else "")
    
        # Customize grid
        plt.grid(True, linestyle='--', alpha=0.2, color='#f8f8f2')
        
        # Add labels and title with custom styling
        plt.xlabel('Feature 1', fontsize=12, color='#f8f8f2', fontweight='bold')
        plt.ylabel('Feature 2', fontsize=12, color='#f8f8f2', fontweight='bold')
        plt.title(f'{dataset_type} Data - Predictions (Accuracy: {accuracy:.4f})', 
                 fontsize=14, 
                 color='#f8f8f2',
                 pad=20,
                 fontweight='bold')
        
        # Improve legend
        handles, labels = plt.gca().get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        legend = plt.legend(*zip(*unique), 
                           bbox_to_anchor=(1.05, 1), 
                           loc='upper left',
                           frameon=True,
                           facecolor='#282a36',
                           edgecolor='#f8f8f2')
        plt.setp(legend.get_texts(), color='#f8f8f2')
        
        # Adjust layout
        plt.tight_layout()
        
        # Create directory if it doesn't exist
        os.makedirs('bpneat_viz', exist_ok=True)
        
        # Add prefix for best network visualization
        prefix = "best_" if is_best else ""
        filename = f"{prefix}{dataset_type}_plot_{gen_num}.png"
        
        # Save with higher DPI and transparent background
        plt.savefig(f'bpneat_viz/{filename}', 
                    dpi=300, 
                    bbox_inches='tight',
                    facecolor='#282a36',
                    edgecolor='none')
        plt.close()

    # Plot for train and test data
    plot_data(train_x, train_preds, train_y, 'Train', gen_num, train_acc)
    plot_data(test_x, test_preds, test_y, 'Test', gen_num, test_acc)

class GenePool:
    def __init__(self):
        self.organisms = []
        for i in range(init_pop):
            self.organisms.append(Genome())
        self.species = [Species(self.organisms, copy.deepcopy(random.choice(self.organisms)))]
        self.best_fitness = float('-inf')
        self.best_org_info = None  # Will store (graph, bools, acts, fitness)
        self.best_generation = None  # Track generation of best fitness

    def log(self, gen_num, x, y, thegraphs, thebools, theacts):
        i, org = max(enumerate(self.organisms), key=lambda org: org[1].fitness)
        print(f"Top fitness in generation {gen_num}: {org.fitness}")

        win_graph = copy.deepcopy(thegraphs[i])

        orig_indices = np.arange(MAX_NODE_CT)
        orig_indices[:len(org.toposort)] = org.toposort
        mapping = np.argsort(orig_indices)

        win_graph = win_graph[mapping]
        win_graph = np.array([row[mapping] for row in win_graph])

        # Update best organism if current one is better
        if org.fitness > self.best_fitness:
            self.best_fitness = org.fitness
            self.best_org_info = (win_graph, thebools[i], org.activations, org.fitness)
            self.best_generation = gen_num  # Update best generation
            
            # Save best visualization with special name
            visualize_graph(win_graph, gen_num, org.fitness, org.activations, is_best=True)
            test_viz(thegraphs[i], thebools[i], theacts[i], x, y, gen_num, is_best=True)

        # Save regular visualization
        visualize_graph(win_graph, gen_num, org.fitness, org.activations)
        test_viz(thegraphs[i], thebools[i], theacts[i], x, y, gen_num)
        
        return org.fitness

    def new_gen(self):
        # Existing code remains the same
        Genome.mutation = dict()
        for s in self.species:
            s.members = []

        self.organisms = [org for org in self.organisms if not math.isnan(org.fitness)]

        for org in self.organisms:
            org.toposort = topoSort(org.graph)
            for s in self.species:
                if s.is_member(org):
                    break
            else:
                self.species.append(Species(org))

        self.species = [s for s in self.species if len(s.members) > 0 and s.stagnation_num < stagnation_cutoff and not math.isnan(s.fitness)]

        print("num species: ", len(self.species))

        summed_fitness = sum(s.fitness for s in self.species)
        ct = 0

        for s in self.species:
            member_ct = int((s.fitness * init_pop) / summed_fitness)
            if member_ct > 0:
                s.new_gen_spec(member_ct)
                ct += len(s.members)

        self.organisms = []

        for s in self.species:
            self.organisms += s.members
        print("population: ", len(self.organisms))

    def calculate_complexity_penalty(self, org, base_nodes=None):
        """
        Calculate complexity penalty considering:
        1. Number of excess nodes beyond base architecture
        2. Number of connections
        3. Activation function diversity
        """
        if base_nodes is None:
            base_nodes = num_in + num_out
            
        # Node complexity - penalize excess nodes
        excess_nodes = org.node_ct - base_nodes
        node_penalty = math.log(1 + excess_nodes)
        
        # Connection density penalty
        possible_connections = org.node_ct * org.node_ct
        density = org.connection_ct / possible_connections if possible_connections > 0 else 0
        connection_penalty = density * math.log(1 + org.connection_ct)
        
        # Activation diversity penalty
        unique_activations = len(set(act for act in org.activations if act > 0))  # Ignore default 0 activations
        activation_penalty = math.log(1 + unique_activations)
        
        # Combine penalties with weights
        total_penalty = (
            0.3 * node_penalty +      # Weight for node complexity
            0.5 * connection_penalty + # Weight for connection complexity
            0.2 * activation_penalty   # Weight for activation complexity
        )
        
        return 1 + total_penalty

    def eval(self, x, y):
        print("starting eval")
        x, y = jnp.array(x[:len(x)//2]), jnp.array(y[:len(y)//2])
        the_graphs, the_bools, the_activations = population_stack(self.organisms)
        fitnesses, the_graphs = eval_orgs(the_graphs, the_bools, the_activations, x, y)
        
        for idx, org in enumerate(self.organisms):
            # Apply the enhanced complexity penalty
            complexity_penalty = self.calculate_complexity_penalty(org)
            org.fitness = -fitnesses[idx] / complexity_penalty
        
        for s in self.species:
            s.update_fitness()
        return np.array(the_graphs), the_bools, the_activations


    def save_best_network(self, problem_name):
        if self.best_org_info is not None:
            graph, bools, acts, fitness = self.best_org_info
            
            # Save network data
            best_data = {
                'graph': graph,
                'bools': bools,
                'activations': acts,
                'fitness': fitness
            }
            
            # Create directory if it doesn't exist
            os.makedirs('best_networks', exist_ok=True)
            
            # Save using numpy's save function
            save_path = os.path.join('best_networks', f'best_network_{problem_name.lower()}.npz')
            np.savez(save_path, **best_data)
            print(f"Saved best network for {problem_name} with fitness {fitness}")

if __name__ == "__main__":
    # List of all problems to test
    problems = [
        ('XOR', generate_xor),
        ('Circle', generate_circle),
        ('Spiral', generate_spiral)
    ]
    
    # Create base directories
    base_dirs = ['bpneat_graphs', 'bpneat_viz', 'best_networks']
    for dir_name in base_dirs:
        os.makedirs(dir_name, exist_ok=True)
    
    # Run experiments for each problem
    for problem_name, generator in problems:
        print(f"\n{'='*50}")
        print(f"Starting experiment for {problem_name} problem")
        print(f"{'='*50}")
        
        # Generate data
        x, y = generator(400)
        print(f"Generated data for {problem_name}")
        print(f"Input shape: {x.shape}, Labels shape: {y.shape}")
        
        # Create output directories for this problem
        problem_graphs_dir = f"bpneat_graphs_{problem_name.lower()}"
        problem_viz_dir = f"bpneat_viz_{problem_name.lower()}"
        problem_best_dir = f"best_networks_{problem_name.lower()}"
        
        os.makedirs(problem_graphs_dir, exist_ok=True)
        os.makedirs(problem_viz_dir, exist_ok=True)
        os.makedirs(problem_best_dir, exist_ok=True)
        
        # Initialize new population for this problem
        pop = GenePool()
        ct = 0
        
        _ = input(f"Press Enter to start evolution for {problem_name}...")
        
        while ct < num_generations:
            print(f"\nGeneration {ct + 1}/{num_generations}")
            thegraphs, thebools, theacts = pop.eval(x, y)
            ct += 1
            current_fitness = pop.log(ct, x, y, thegraphs, thebools, theacts)
            print(f"Current max fitness: {current_fitness}")
            print(f"Best fitness so far: {pop.best_fitness} (Generation {pop.best_generation})")
            pop.new_gen()
            
        print(f"\nFinished {num_generations} generations for {problem_name}")
        print(f"Final best fitness: {pop.best_fitness} achieved in generation {pop.best_generation}")
        
        # Save the best network
        pop.save_best_network(problem_name)
        
        # Move files to problem-specific directories if directories exist
        for src_dir, dest_dir in [
            ("bpneat_graphs", problem_graphs_dir),
            ("bpneat_viz", problem_viz_dir),
            ("best_networks", problem_best_dir)
        ]:
            if os.path.exists(src_dir) and os.path.exists(dest_dir):
                for filename in os.listdir(src_dir):
                    src_file = os.path.join(src_dir, filename)
                    dest_file = os.path.join(dest_dir, filename)
                    try:
                        shutil.move(src_file, dest_file)
                    except Exception as e:
                        print(f"Error moving {filename} from {src_dir} to {dest_dir}: {e}")
        
        # Copy the best network visualization to a special location
        best_gen = pop.best_generation
        if best_gen is not None:
            for suffix in ['png']:
                try:
                    source_file = os.path.join(problem_graphs_dir, f"best_{best_gen}.{suffix}")
                    dest_file = os.path.join(problem_best_dir, f"best_network_{problem_name.lower()}.{suffix}")
                    if os.path.exists(source_file):
                        shutil.copy2(source_file, dest_file)
                except Exception as e:
                    print(f"Error copying best network file: {e}")
