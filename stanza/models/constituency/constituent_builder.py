
import torch
import torch.nn as nn

from stanza.models.constituency.parse_transitions import Constituent
from stanza.models.constituency.parse_tree import Tree
from stanza.models.constituency.utils import build_nonlinearity

class ConstituentBuilder(nn.Module):
    def __init__(self, open_nodes, args):
        super().__init__()

        self.hidden_size = args['hidden_size']

        self.open_nodes = sorted(list(open_nodes))
        # an embedding for the spot on the constituent LSTM taken up by the Open transitions
        # the pattern when condensing constituents is embedding - con1 - con2 - con3 - embedding
        # TODO: try the two ends have different embeddings?
        self.open_node_map = { x: i for (i, x) in enumerate(self.open_nodes) }
        self.open_node_embedding = nn.Embedding(num_embeddings = len(self.open_node_map),
                                                embedding_dim = self.hidden_size)
        self.register_buffer('open_node_tensors', torch.tensor(range(len(open_nodes)), requires_grad=False))
        self.lstm_input_dropout = nn.Dropout(args['lstm_input_dropout'])

        self.num_layers = args['num_lstm_layers']
        self.lstm_layer_dropout = args['lstm_layer_dropout']

        # forward and backward pieces for crunching several
        # constituents into one, combined into a bi-lstm
        # TODO: make the hidden size here an option?
        self.constituent_reduce_lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers, bidirectional=True, dropout=self.lstm_layer_dropout)
        # affine transformation from bi-lstm reduce to a new hidden layer
        self.reduce_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)
        if args['nonlinearity'] in ('relu', 'leaky_relu'):
            nn.init.kaiming_normal_(self.reduce_linear.weight, nonlinearity=args['nonlinearity'])
            nn.init.uniform_(self.reduce_linear.bias, 0, 1 / (self.hidden_size * 2) ** 0.5)

        self.nonlinearity = build_nonlinearity(args['nonlinearity'])

    def build_constituents(self, labels, children_lists):
        """
        labels: a list of the top labels to produce
        children_lists: a list of lists
          each sublist has the children being created by this function call
        """
        label_hx = [self.open_node_embedding(self.open_node_tensors[self.open_node_map[label]]) for label in labels]

        max_length = max(len(children) for children in children_lists)
        zeros = torch.zeros(self.hidden_size, device=label_hx[0].device)
        node_hx = [[child.output for child in children] for children in children_lists]
        # weirdly, this is faster than using pack_sequence
        unpacked_hx = [[lhx] + nhx + [lhx] + [zeros] * (max_length - len(nhx)) for lhx, nhx in zip(label_hx, node_hx)]
        unpacked_hx = [self.lstm_input_dropout(torch.stack(nhx)) for nhx in unpacked_hx]
        packed_hx = torch.stack(unpacked_hx, axis=1)
        packed_hx = torch.nn.utils.rnn.pack_padded_sequence(packed_hx, [len(x)+2 for x in children_lists], enforce_sorted=False)
        lstm_output = self.constituent_reduce_lstm(packed_hx)
        # take just the output of the final layer
        #   result of lstm is ouput, (hx, cx)
        #   so [1][0] gets hx
        #      [1][0][-1] is the final output
        # will be shape len(children_lists) * 2, hidden_size for bidirectional
        # where forward outputs are -2 and backwards are -1
        lstm_output = lstm_output[1][0]
        forward_hx = lstm_output[-2, :]
        backward_hx = lstm_output[-1, :]

        hx = self.reduce_linear(torch.cat((forward_hx, backward_hx), axis=1))
        hx = self.nonlinearity(hx)

        constituents = []
        for idx, (label, children) in enumerate(zip(labels, children_lists)):
            children = [child.value for child in children]
            if isinstance(label, str):
                node = Tree(label=label, children=children)
            else:
                for value in reversed(label):
                    node = Tree(label=value, children=children)
                    children = node
            constituents.append(Constituent(value=node, hx=hx[idx, :]))
        return constituents
        
