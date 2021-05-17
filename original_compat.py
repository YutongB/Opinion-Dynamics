from original.read_graph import remove_files
import numpy as np

PREFIX = "output/original"

def write_prior(self, path_prefix=PREFIX):
    # save the new prior
    for i in self.g:
        with open(f'{path_prefix}/data/node{i}.txt', 'a') as f:
            for data in self.g.nodes[i]['prior']:
                f.write(str(data) + ' ')
            f.write('\n')

def save_graph_priors(n, initial_distr, final_distr, path_prefix=PREFIX):
    # save the priors into a new text file
    remove_files(f'{path_prefix}/data', starts_with='node', ends_with='.txt')

    sims = len(initial_distr)

    for node in range(n):
        with open(f'{path_prefix}/data/node{node}.txt', 'w+') as f:
            def write_prior(prior):
                for i in prior:
                    f.write(str(i) + ' ')
                f.write('\n')

            for sim in range(sims):
                write_prior(initial_distr[sim][node])
                write_prior(final_distr[sim][node])        


def save_graph(n, friendliness, path_prefix=PREFIX):
    # save information about the initial graph
    with open(f'{path_prefix}/data/graph.txt', 'w+') as f:
        f.write('Number of nodes: {}\n'.format(n))

        for ix,iy in np.ndindex(friendliness.shape):
            f.write('{} {} {}\n'.format(ix, iy, friendliness[ix,iy]))


def save_tosses(true_bias, heads_list, tosses_list, path_prefix=PREFIX):
    with open(f'{path_prefix}/data/observations.txt', 'w+') as f:
        f.write('True coin bias: {}\n'.format(true_bias))
        f.write('Iteration Num_heads Num_tosses\n')
        for i, heads, tosses in enumerate(zip(heads_list, tosses_list)):
            f.write(f'{i+1} {heads} {tosses}\n')

