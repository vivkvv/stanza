"""
A version of the BaseModel which uses LSTMs to predict the correct next transition
based on the current known state.

The primary purpose of this class is to implement the prediction of the next
transition, which is done by concatenating the output of an LSTM operated over
previous transitions, the words, and the partially built constituents.
"""

import logging
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from stanza.models.constituency.base_model import BaseModel
from stanza.models.constituency.constituent_builder import ConstituentBuilder
from stanza.models.constituency.parse_transitions import TransitionScheme, TransitionNode, Constituent, ConstituentNode
from stanza.models.constituency.parse_tree import Tree
from stanza.models.constituency.tree_stack import TreeStack
from stanza.models.constituency.utils import build_nonlinearity
from stanza.models.constituency.word_embedder import WordEmbedder

logger = logging.getLogger('stanza')

class LSTMModel(BaseModel, nn.Module):
    def __init__(self, pt, forward_charlm, backward_charlm, transitions, constituents, tags, words, rare_words, root_labels, open_nodes, unary_limit, args):
        """
        pt: a Pretrain object
        transitions: a list of all possible transitions which will be
          used to build trees
        constituents: a list of all possible constituents in the treebank
        tags: a list of all possible tags in the treebank
        words: a list of all known words, used for a delta word embedding.
          note that there will be an attempt made to learn UNK words as well,
          and tags by themselves may help UNK words
        rare_words: a list of rare words, used to occasionally replace with UNK
        root_labels: probably ROOT, although apparently some treebanks like TOP
        open_nodes: a list of all possible open nodes which will go on the stack
          - this might be different from constituents if there are nodes
            which represent multiple constituents at once
        args: hidden_size, transition_hidden_size, etc as gotten from
          constituency_parser.py

        Note that it might look like a hassle to pass all of this in
        when it can be collected directly from the trees themselves.
        However, that would only work at train time.  At eval or
        pipeline time we will load the lists from the saved model.
        """
        super().__init__()
        self.args = args
        self.unsaved_modules = []

        self.add_unsaved_module('embedding', nn.Embedding.from_pretrained(torch.from_numpy(pt.emb), freeze=True))

        if forward_charlm is not None:
            self.add_unsaved_module('forward_charlm', forward_charlm)
            self.add_unsaved_module('forward_charlm_vocab', forward_charlm.char_vocab())
        else:
            self.forward_charlm = None
            self.forward_charlm_vocab = None
        if backward_charlm is not None:
            self.add_unsaved_module('backward_charlm', backward_charlm)
            self.add_unsaved_module('backward_charlm_vocab', backward_charlm.char_vocab())
        else:
            self.backward_charlm = None
            self.backward_charlm_vocab = None

        self.word_embedder = WordEmbedder(pt, forward_charlm, backward_charlm, tags, words, rare_words, args)
        self.constituent_builder = ConstituentBuilder(open_nodes, args)

        self.root_labels = sorted(list(root_labels))
        self.constituents = sorted(list(constituents))
        self.constituent_map = { x: i for (i, x) in enumerate(self.constituents) }
        # precompute tensors for the constituents
        self.register_buffer('constituent_tensors', torch.tensor(range(len(self.constituent_map)), requires_grad=False))

        self.hidden_size = self.args['hidden_size']
        self.transition_hidden_size = self.args['transition_hidden_size']
        self.transition_embedding_dim = self.args['transition_embedding_dim']

        self.transitions = sorted(list(transitions))
        self.transition_map = { t: i for i, t in enumerate(self.transitions) }
        # precompute tensors for the transitions
        self.register_buffer('transition_tensors', torch.tensor(range(len(transitions)), requires_grad=False))
        self.transition_embedding = nn.Embedding(num_embeddings = len(transitions),
                                                 embedding_dim = self.transition_embedding_dim)

        self.num_layers = self.args['num_lstm_layers']
        self.lstm_layer_dropout = self.args['lstm_layer_dropout']

        # also register a buffer of zeros so that we can always get zeros on the appropriate device
        self.register_buffer('transition_zeros', torch.zeros(self.num_layers, 1, self.transition_hidden_size))
        self.register_buffer('constituent_zeros', torch.zeros(self.num_layers, 1, self.hidden_size))

        self.transition_lstm = nn.LSTM(input_size=self.transition_embedding_dim, hidden_size=self.transition_hidden_size, num_layers=self.num_layers, dropout=self.lstm_layer_dropout)
        # input_size is hidden_size - could introduce a new constituent_size instead if we liked
        self.constituent_lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=self.lstm_layer_dropout)

        self._transition_scheme = args['transition_scheme']
        if self._transition_scheme is TransitionScheme.TOP_DOWN_UNARY:
            unary_transforms = {}
            for constituent in self.constituent_map:
                unary_transforms[constituent] = nn.Linear(self.hidden_size, self.hidden_size)
            self.unary_transforms = nn.ModuleDict(unary_transforms)

        self.open_nodes = self.constituent_builder.open_nodes
        self.open_node_map = self.constituent_builder.open_node_map
        self.register_buffer('open_node_tensors', self.constituent_builder.open_node_tensors)
        # TODO: remove this `get` once it's not needed
        if args.get('combined_dummy_embedding', False):
            self.dummy_embedding = self.constituent_builder.open_node_embedding
        else:
            self.dummy_embedding = nn.Embedding(num_embeddings = len(self.open_node_map),
                                                embedding_dim = self.hidden_size)
        self.register_buffer('open_node_tensors', torch.tensor(range(len(open_nodes)), requires_grad=False))

        self.nonlinearity = build_nonlinearity(self.args['nonlinearity'])

        self.predict_dropout = nn.Dropout(self.args['predict_dropout'])
        self.lstm_input_dropout = nn.Dropout(self.args['lstm_input_dropout'])

        # matrix for predicting the next transition using word/constituent/transition queues
        # word size + constituency size + transition size
        middle_layers = self.args['num_output_layers'] - 1
        predict_input_size = [self.hidden_size * 2 + self.transition_hidden_size] + [self.hidden_size] * middle_layers
        predict_output_size = [self.hidden_size] * middle_layers + [len(transitions)]
        self.output_layers = nn.ModuleList([nn.Linear(input_size, output_size)
                                            for input_size, output_size in zip(predict_input_size, predict_output_size)])
        for output_layer, input_size in zip(self.output_layers, predict_input_size):
            if self.args['nonlinearity'] in ('relu', 'leaky_relu'):
                nn.init.kaiming_normal_(output_layer.weight, nonlinearity=self.args['nonlinearity'])
                nn.init.uniform_(output_layer.bias, 0, 1 / input_size ** 0.5)

        self.constituency_lstm = self.args['constituency_lstm']

        self._unary_limit = unary_limit

    def num_words_known(self, words):
        return sum(word in self.word_embedder.vocab_map or word.lower() in self.word_embedder.vocab_map for word in words)

    def add_unsaved_module(self, name, module):
        """
        Adds a module which will not be saved to disk

        Best used for large models such as pretrained word embeddings
        """
        self.unsaved_modules += [name]
        setattr(self, name, module)

    def get_root_labels(self):
        return self.root_labels

    def initial_word_queues(self, tagged_word_lists):
        """
        Produce initial word queues out of the model's LSTMs for use in the tagged word lists.

        Operates in a batched fashion to reduce the runtime for the LSTM operations
        """
        return self.word_embedder.initial_word_queues(tagged_word_lists, self.embedding, self.forward_charlm, self.forward_charlm_vocab, self.backward_charlm, self.backward_charlm_vocab)

    def initial_transitions(self):
        """
        Return an initial TreeStack with no transitions
        """
        return TreeStack(value=TransitionNode(None, self.transition_zeros[-1, 0, :], self.transition_zeros, self.transition_zeros), parent=None, length=1)

    def initial_constituents(self):
        """
        Return an initial TreeStack with no constituents
        """
        return TreeStack(value=ConstituentNode(None, self.constituent_zeros[-1, 0, :], self.constituent_zeros, self.constituent_zeros), parent=None, length=1)

    def get_word(self, word_node):
        return word_node.value

    def transform_word_to_constituent(self, state):
        word_node = state.word_queue[state.word_position]
        word = word_node.value
        return Constituent(value=word, hx=word_node.hx)

    def dummy_constituent(self, dummy):
        label = dummy.label
        open_index = self.open_node_tensors[self.open_node_map[label]]
        hx = self.dummy_embedding(open_index)
        return Constituent(value=dummy, hx=hx)

    def unary_transform(self, constituents, labels):
        top_constituent = constituents.value
        node = top_constituent.value
        hx = top_constituent.output
        for label in reversed(labels):
            node = Tree(label=label, children=[node])
            hx = self.unary_transforms[label](hx)
            # non-linearity after the unary transform
            hx = self.nonlinearity(hx)
        top_constituent = Constituent(value=node, hx=hx)
        return top_constituent

    def build_constituents(self, labels, children_lists):
        """
        labels: a list of the top labels to produce
        children_lists: a list of lists
          each sublist has the children being created by this function call
        """
        return self.constituent_builder.build_constituents(labels, children_lists)

    def push_constituents(self, constituent_stacks, constituents):
        current_nodes = [stack.value for stack in constituent_stacks]

        constituent_input = torch.stack([x.hx for x in constituents])
        constituent_input = constituent_input.unsqueeze(0)
        constituent_input = self.lstm_input_dropout(constituent_input)

        hx = torch.cat([current_node.hx for current_node in current_nodes], axis=1)
        cx = torch.cat([current_node.cx for current_node in current_nodes], axis=1)
        output, (hx, cx) = self.constituent_lstm(constituent_input, (hx, cx))
        if self.constituency_lstm:
            new_stacks = [stack.push(ConstituentNode(constituent.value, output[0, i, :], hx[:, i:i+1, :], cx[:, i:i+1, :]))
                          for i, (stack, constituent) in enumerate(zip(constituent_stacks, constituents))]
        else:
            new_stacks = [stack.push(ConstituentNode(constituent.value, constituents[i].hx, hx[:, i:i+1, :], cx[:, i:i+1, :]))
                          for i, (stack, constituent) in enumerate(zip(constituent_stacks, constituents))]
        return new_stacks

    def get_top_constituent(self, constituents):
        """
        Extract only the top constituent from a state's constituent
        sequence, even though it has multiple addition pieces of
        information
        """
        constituent_node = constituents.value
        return constituent_node.value

    def push_transitions(self, transition_stacks, transitions):
        transition_idx = torch.stack([self.transition_tensors[self.transition_map[transition]] for transition in transitions])
        transition_input = self.transition_embedding(transition_idx).unsqueeze(0)
        transition_input = self.lstm_input_dropout(transition_input)

        hx = torch.cat([t.value.hx for t in transition_stacks], axis=1)
        cx = torch.cat([t.value.cx for t in transition_stacks], axis=1)
        output, (hx, cx) = self.transition_lstm(transition_input, (hx, cx))
        new_stacks = [stack.push(TransitionNode(transition, output[0, i, :], hx[:, i:i+1, :], cx[:, i:i+1, :]))
                      for i, (stack, transition) in enumerate(zip(transition_stacks, transitions))]
        return new_stacks

    def get_top_transition(self, transitions):
        """
        Extract only the top transition from a state's transition
        sequence, even though it has multiple addition pieces of
        information
        """
        transition_node = transitions.value
        return transition_node.value

    def unary_limit(self):
        return self._unary_limit

    def transition_scheme(self):
        return self._transition_scheme

    def has_unary_transitions(self):
        return self._transition_scheme is TransitionScheme.TOP_DOWN_UNARY

    def is_top_down(self):
        return self._transition_scheme in (TransitionScheme.TOP_DOWN, TransitionScheme.TOP_DOWN_UNARY, TransitionScheme.TOP_DOWN_COMPOUND)

    def forward(self, states):
        """
        Return logits for a prediction of what transition to make next

        We've basically done all the work analyzing the state as
        part of applying the transitions, so this method is very simple
        """
        word_hx = torch.stack([state.word_queue[state.word_position].hx for state in states])
        transition_hx = torch.stack([state.transitions.value.output for state in states])
        # note that we use hx instead of output from the constituents
        # this way, we can, as an option, NOT include the constituents to the left
        # when building the current vector for a constituent
        # and the vector used for inference will still incorporate the entire LSTM
        constituent_hx = torch.stack([state.constituents.value.hx[-1, 0, :] for state in states])

        hx = torch.cat((word_hx, transition_hx, constituent_hx), axis=1)
        for idx, output_layer in enumerate(self.output_layers):
            hx = self.predict_dropout(hx)
            if idx < len(self.output_layers) - 1:
                hx = self.nonlinearity(hx)
            hx = output_layer(hx)
        return hx

    # TODO: merge this with forward?
    def predict(self, states, is_legal=False):
        """
        Generate and return predictions, along with the transitions those predictions represent

        If is_legal is set to True, will only return legal transitions.
        This means returning None if there are no legal transitions.
        Hopefully the constraints prevent that from happening
        """
        predictions = self.forward(states)
        pred_max = torch.argmax(predictions, axis=1)

        pred_trans = [self.transitions[pred_max[idx]] for idx in range(len(states))]
        if is_legal:
            for idx, (state, trans) in enumerate(zip(states, pred_trans)):
                if not trans.is_legal(state, self):
                    _, indices = predictions[idx, :].sort(descending=True)
                    for index in indices:
                        if self.transitions[index].is_legal(state, self):
                            pred_trans[idx] = self.transitions[index]
                            break
                    else: # yeah, else on a for loop, deal with it
                        pred_trans[idx] = None

        return predictions, pred_trans

    def get_params(self):
        """
        Get a dictionary for saving the model
        """
        model_state = self.state_dict()

        # skip saving modules like pretrained embeddings, because they are large and will be saved in a separate file
        skipped = [k for k in model_state.keys() if k.split('.')[0] == 'word_embedder']
        for k in skipped:
            del model_state[k]

        params = {
            'model': model_state,
            'model_type': "LSTM",
            'config': self.args,
            'transitions': self.transitions,
            'constituents': self.constituents,
            'tags': self.word_embedder.tags,
            'words': self.word_embedder.delta_words,
            'rare_words': self.word_embedder.rare_words,
            'root_labels': self.root_labels,
            'open_nodes': self.open_nodes,
        }

        return params

