import torch
from torch import nn

from stanza.models.constituency.constituent_builder import ConstituentBuilder
from stanza.models.constituency.parse_transitions import Constituent, ConstituentNode
from stanza.models.constituency.parse_tree import Tree
from stanza.models.constituency.utils import build_nonlinearity
from stanza.models.constituency.word_embedder import WordEmbedder

class LSTMDiscriminator(nn.Module):
    def __init__(self, pt, forward_charlm, backward_charlm, transitions, tags, words, rare_words, open_nodes, args):
        super().__init__()

        self.args = args
        self.unsaved_modules = []

        # TODO: can all this be factored out with lstm_model?
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

        self.hidden_size = args['hidden_size']
        middle_layers = args['num_output_layers'] - 1
        predict_input_size = [self.hidden_size] * (middle_layers + 1)
        # contract to a single value at the end
        predict_output_size = [self.hidden_size] * middle_layers + [1]
        self.output_layers = nn.ModuleList([nn.Linear(input_size, output_size)
                                            for input_size, output_size in zip(predict_input_size, predict_output_size)])

        self.nonlinearity = build_nonlinearity(self.args['nonlinearity'])
        self.predict_dropout = nn.Dropout(self.args['predict_dropout'])

    def output_from_embedding(self, hx):
        """
        Go from the embedding of the final constituent to an output

        Output represents the logit of whether or not the tree is an original tree.
        Is not already normalized
        """
        if isinstance(hx, list):
            if isinstance(hx[0], Constituent):
                hx = [x.hx for x in hx]
            hx = torch.stack(hx, axis=0)
        for idx, output_layer in enumerate(self.output_layers):
            hx = self.predict_dropout(hx)
            if idx < len(self.output_layers) - 1:
                hx = self.nonlinearity(hx)
            hx = output_layer(hx)
        return hx

    def add_unsaved_module(self, name, module):
        """
        Adds a module which will not be saved to disk

        Best used for large models such as pretrained word embeddings
        """
        self.unsaved_modules += [name]
        setattr(self, name, module)

    def recursive_tree_embedding(self, tree, embedding, embedding_index):
        if tree.is_leaf() or tree.is_preterminal():
            return embedding[embedding_index], embedding_index + 1
        children = []
        for child in tree.children:
            child_embedding, embedding_index = self.recursive_tree_embedding(child, embedding, embedding_index)
            children.append(ConstituentNode(None, child_embedding.hx, None, None))
        recursive_embedding = self.constituent_builder.build_constituents((tree.label,), (children,))[0]
        return recursive_embedding, embedding_index

    def forward(self, trees):
        preterminal_lists = [[Tree(label=pt.label, children=Tree(label=pt.children[0].label))
                              for pt in tree.yield_preterminals()]
                             for tree in trees]
        embeddings = self.word_embedder.initial_word_queues(preterminal_lists, self.embedding, self.forward_charlm, self.forward_charlm_vocab, self.backward_charlm, self.backward_charlm_vocab)

        # TODO: we can unwind this recursion
        # unwinding the recursion will let us batch operations (eg, significantly faster)
        full_tree_embeddings = [self.recursive_tree_embedding(tree, embedding, 0)[0] for tree, embedding in zip(trees, embeddings)]

        result = self.output_from_embedding(full_tree_embeddings)
        return result

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
        }
        return params
