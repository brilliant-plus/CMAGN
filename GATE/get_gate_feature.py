import tensorflow._api.v2.compat.v1 as tf
import matplotlib.pyplot as plt

from data_process import prepare_graph_data, parse_args
from GATE.gate_trainer import GATETrainer


def get_gate_feature(net, fea, epochs, l):
    args = parse_args(epochs=epochs, l=l)
    feature_dim = fea.shape[1]
    args.hidden_dims = [feature_dim] + args.hidden_dims

    G, S, R = prepare_graph_data(net)
    gate_trainer = GATETrainer(args)
    gate_trainer(G, fea, S, R)
    embeddings, attention = gate_trainer.infer(G, fea, S, R)
    tf.reset_default_graph()
    return embeddings
