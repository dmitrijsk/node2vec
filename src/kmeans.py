import argparse
import os
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.cluster import KMeans
from matplotlib import cm


def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser")

    parser.add_argument(
        "--demo",
        help="Run demo. \
              Options: les_miserables_homophily, les_miserables_str_eq_node2vec, les_miserables_str_eq_struc2vec",
        default="")

    parser.add_argument(
        "--path_to_edgelist",
        help="Path to the *.edgelist file that represents the network.",
        default="")

    parser.add_argument(
        "--edgelist_sep",
        help="Separator used in the edgelist file. Default: space",
        default=" ")

    parser.add_argument(
        "--p",
        help="Return parameter. Default: 1",
        default=1,
        type=float)

    parser.add_argument(
        "--q",
        help="In-out parameter. Default: 0.5",
        default=0.5,
        type=float)

    parser.add_argument(
        "--d",
        help="Dimensionality of embeddings. Default: 16",
        default=16,
        type=int)

    parser.add_argument(
        "--k",
        help="Window (context) size. Default: 10",
        default=10,
        type=int)

    parser.add_argument(
        "--l",
        help="Walk length. Default: 80",
        default=80,
        type=int)

    parser.add_argument(
        "--n_clusters",
        help="Number of k-means clusters. Default: 6",
        default=6,
        type=int)

    parser.add_argument(
        "--plot_labels",
        help="Plot node labels. Default: false",
        action='store_true')

    return parser.parse_args()


def update_args(args):
    """Set arguments for demos."""

    args.params = f"d_{args.d}_l_{args.l}_k_{args.k}_p_{args.p}_q_{args.q}"
    args.data = os.path.basename(args.path_to_edgelist).split(".")[0]
    args.emb_filename = f"emb/{args.data}/{args.data}_{args.params}.emb"

    if args.demo == "les_miserables_homophily":
        args.path_to_edgelist = "graph/les_miserables/les_miserables.edgelist"
        args.p, args.q, args.n_clusters = 1, 0.5, 6
        args.cmap = ["#d7191c", "#d4e4bd", "#e7745d", "#f7d09e", "#2c7bb6", "#80afb9"]
        args.params = f"d_{args.d}_l_{args.l}_k_{args.k}_p_{args.p}_q_{args.q}"
        args.data = os.path.basename(args.path_to_edgelist).split(".")[0]
        args.emb_filename = f"emb/{args.data}/{args.data}_{args.params}.emb"

    elif args.demo == "les_miserables_str_eq_node2vec":
        args.path_to_edgelist = "graph/les_miserables/les_miserables.edgelist"
        args.p, args.q, args.n_clusters = 1, 2, 3
        args.cmap = ["#2c7bb6", "#d7191c", "#e9ebb2"]
        args.params = f"d_{args.d}_l_{args.l}_k_{args.k}_p_{args.p}_q_{args.q}"
        args.data = os.path.basename(args.path_to_edgelist).split(".")[0]
        args.emb_filename = f"emb/{args.data}/{args.data}_{args.params}.emb"

    elif args.demo == "les_miserables_str_eq_struc2vec":
        args.path_to_edgelist = "graph/les_miserables/les_miserables.edgelist"
        args.n_clusters = 4
        args.cmap = ["#2c7bb6", "#d7191c", "#e9ebb2", "#d7191c"]
        args.params = f"d_16_l_80_k_10_struc2vec"
        args.data = os.path.basename(args.path_to_edgelist).split(".")[0]
        # struc2vec embeddings were created using https://github.com/xiangyue9607/BioNEV.
        # Commands:
        # cd BioNEV
        # bionev --input ../node2vec/graph/les_miserables.edgelist \
        #        --output ../node2vec/emb/les_miserables_struc2vec.emb \
        #        --method struc2vec
        #        --task link-prediction
        #        --walk-length 80
        #        --window-size 10
        #        --dimensions 16
        args.emb_filename = "emb/les_miserables/les_miserables_struc2vec.emb"

    return args


def save_fig(filename, h, w, dpi):
    """Save the plot with specified height, width and dpi parameters."""

    figure = plt.gcf()
    figure.set_size_inches(w, h)
    plt.savefig(filename, dpi=dpi)


def generate_node2vec_embeddings(args):
    """Execute node2vec to create embeddings."""

    if args.demo != "les_miserables_str_eq_struc2vec":
        cmd = "python src/main.py " + \
              "--input " + args.path_to_edgelist + \
              " --output " + args.emb_filename + \
              " --dimensions " + str(args.d) + \
              " --walk-length " + str(args.l) + \
              " --window-size " + str(args.k) + \
              " --p " + str(args.p) + \
              " --q " + str(args.q)

        print(f"Command: {cmd}")
        os.system(cmd)


def read_embeddings(args, graph):
    """Read embeddings from an external file."""

    with open(args.emb_filename) as f:
        # Ignore the first line (# of nodes, # of dimensions).
        emb = f.read().splitlines()[1:]

    emb = [e.split() for e in emb]  # Split with whitespace.

    node_names = list(graph.nodes())  # Used to replace integers with character names.
    emb = {node_names[int(e[0])]: [float(ee) for ee in e[1:]] for e in emb}  # Convert embeddings to float.

    emb_lst = list(emb.values())
    return emb, emb_lst, node_names


def plot_elbow_method(emb_lst):
    """Plot and save an elbow method for kmeans."""

    # Source: https://stackoverflow.com/questions/19197715/scikit-learn-k-means-elbow-criterion
    sse = {}
    for k in range(1, 30):
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(emb_lst)
        sse[k] = kmeans.inertia_  # Inertia: Sum of distances of samples to their closest cluster center
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of clusters")
    plt.ylabel("SSE")
    plt.show()


def plot_network_with_clusters(emb, kmeans, node_names):
    """Plot the original graph with colours according to the clusters."""

    node_keys = list(emb.keys())
    node_clusters = kmeans.labels_
    if not hasattr(args, 'cmap'):
        args.cmap = [cm.get_cmap('Set1', args.n_clusters)(k) for k in range(args.n_clusters)]
    color_map = []
    for node_name in node_names:
        name_index = node_keys.index(node_name)
        cluster = node_clusters[name_index]
        color_map.append(args.cmap[cluster])

    nx.draw(G,
            node_color=color_map,
            with_labels=args.plot_labels,
            alpha=0.7)
    save_fig(
        f"images/{args.data}/{args.data}_{args.params}_kmeans_{args.n_clusters}_clusters.png",
        h=5,
        w=10,
        dpi=200)
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    # Update args if demonstration is run.
    args = update_args(args)
    # Get network graph data.
    G = nx.readwrite.edgelist.read_edgelist(args.path_to_edgelist, delimiter=args.edgelist_sep)
    # Get embeddings
    generate_node2vec_embeddings(args)
    # Import embeddings.
    emb, emb_lst, node_names = read_embeddings(args, G)
    # k-means clustering with embeddings as features.
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(emb_lst)
    plot_elbow_method(emb_lst)
    # Plot network clusters.
    plot_network_with_clusters(emb, kmeans, node_names)
