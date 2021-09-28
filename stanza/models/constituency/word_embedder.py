from operator import itemgetter
import random

import torch
import torch.nn as nn

from stanza.models.common.data import get_long_tensor
from stanza.models.common.utils import unsort
from stanza.models.common.vocab import PAD_ID, UNK_ID
from stanza.models.constituency.parse_transitions import WordNode
from stanza.models.constituency.utils import build_nonlinearity

class WordEmbedder(nn.Module):
    def __init__(self, pt, forward_charlm, backward_charlm, tags, words, rare_words, args):
        super().__init__()

        # replacing NBSP picks up a whole bunch of words for VI
        self.vocab_map = { word.replace('\xa0', ' '): i for i, word in enumerate(pt.vocab) }
        # precompute tensors for the word indices
        # the tensors should be put on the GPU if needed with a call to cuda()
        self.register_buffer('vocab_tensors', torch.tensor(range(len(pt.vocab)), requires_grad=False))
        self.embedding_dim = pt.emb.shape[1]

        self.tag_embedding_dim = args['tag_embedding_dim']
        self.tags = sorted(list(tags))
        if self.tag_embedding_dim > 0:
            self.tag_map = { t: i+2 for i, t in enumerate(self.tags) }
            self.tag_embedding = nn.Embedding(num_embeddings = len(tags)+2,
                                              embedding_dim = self.tag_embedding_dim,
                                              padding_idx = 0)
            self.register_buffer('tag_tensors', torch.tensor(range(len(self.tags) + 2), requires_grad=False))

        # TODO: add a max_norm?
        self.delta_embedding_dim = args['delta_embedding_dim']
        self.delta_words = sorted(set(words))
        self.delta_word_map = { word: i+2 for i, word in enumerate(self.delta_words) }
        assert PAD_ID == 0
        assert UNK_ID == 1
        self.delta_embedding = nn.Embedding(num_embeddings = len(self.delta_words)+2,
                                            embedding_dim = self.delta_embedding_dim,
                                            padding_idx = 0)
        self.register_buffer('delta_tensors', torch.tensor(range(len(self.delta_words) + 2), requires_grad=False))

        self.rare_words = set(rare_words)
        self.rare_word_unknown_frequency = args['rare_word_unknown_frequency']

        self.word_input_size = self.embedding_dim + self.tag_embedding_dim + self.delta_embedding_dim

        if forward_charlm is not None:
            self.word_input_size += forward_charlm.hidden_dim()
        if backward_charlm is not None:
            self.word_input_size += backward_charlm.hidden_dim()

        # also register a buffer of zeros so that we can always get zeros on the appropriate device
        self.hidden_size = args['hidden_size']
        self.register_buffer('word_zeros', torch.zeros(self.hidden_size))

        self.num_layers = args['num_lstm_layers']
        self.lstm_layer_dropout = args['lstm_layer_dropout']
        self.word_lstm = nn.LSTM(input_size=self.word_input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, bidirectional=True, dropout=self.lstm_layer_dropout)

        # after putting the word_delta_tag input through the word_lstm, we get back
        # hidden_size * 2 output with the front and back lstms concatenated.
        # this transforms it into hidden_size with the values mixed together
        self.word_to_constituent = nn.Linear(self.hidden_size * 2, self.hidden_size)
        if args['nonlinearity'] in ('relu', 'leaky_relu'):
            nn.init.kaiming_normal_(self.word_to_constituent.weight, nonlinearity=args['nonlinearity'])
            nn.init.uniform_(self.word_to_constituent.bias, 0, 1 / (self.hidden_size * 2) ** 0.5)

        self.nonlinearity = build_nonlinearity(args['nonlinearity'])

        self.word_dropout = nn.Dropout(args['word_dropout'])

    def build_char_representation(self, all_word_labels, device, charlm, vocab, forward):
        CHARLM_START = "\n"
        CHARLM_END = " "

        all_data = []
        for idx, word_labels in enumerate(all_word_labels):
            if not forward:
                word_labels = [x[::-1] for x in reversed(word_labels)]

            chars = [CHARLM_START]
            offsets = []
            for w in word_labels:
                chars.extend(w)
                chars.append(CHARLM_END)
                offsets.append(len(chars) - 1)
            if not forward:
                offsets.reverse()
            chars = vocab.map(chars)
            all_data.append((chars, offsets, len(chars), len(all_data)))

        all_data.sort(key=itemgetter(2), reverse=True)
        chars, char_offsets, char_lens, orig_idx = tuple(zip(*all_data))
        chars = get_long_tensor(chars, len(all_data), pad_id=vocab.unit2id(' ')).to(device=device)

        # TODO: surely this should be stuffed in the charlm model itself rather than done here
        with torch.no_grad():
            output, _, _ = charlm.forward(chars, char_lens)
            res = [output[i, offsets] for i, offsets in enumerate(char_offsets)]
            res = unsort(res, orig_idx)

        return res


    def initial_word_queues(self, tagged_word_lists, embedding, forward_charlm, forward_charlm_vocab, backward_charlm, backward_charlm_vocab):
        """
        Produce initial word queues out of the model's LSTMs for use in the tagged word lists.

        Operates in a batched fashion to reduce the runtime for the LSTM operations
        """
        device = next(self.parameters()).device

        vocab_map = self.vocab_map
        def map_word(word):
            idx = vocab_map.get(word, None)
            if idx is not None:
                return idx
            return vocab_map.get(word.lower(), UNK_ID)

        all_word_inputs = []
        all_word_labels = []
        for sentence_idx, tagged_words in enumerate(tagged_word_lists):
            word_ids = [word.children[0].label for word in tagged_words]
            word_idx = torch.stack([self.vocab_tensors[map_word(word.children[0].label)] for word in tagged_words])
            word_input = embedding(word_idx)

            # this occasionally learns UNK at train time
            word_labels = [word.children[0].label for word in tagged_words]
            if self.training:
                delta_labels = [None if word in self.rare_words and random.random() < self.rare_word_unknown_frequency else word
                                for word in word_labels]
            else:
                delta_labels = word_labels
            delta_idx = torch.stack([self.delta_tensors[self.delta_word_map.get(word, UNK_ID)] for word in delta_labels])

            delta_input = self.delta_embedding(delta_idx)

            word_inputs = [word_input, delta_input]

            if self.tag_embedding_dim > 0:
                if self.training:
                    tag_labels = [None if random.random() < self.args['tag_unknown_frequency'] else word.label for word in tagged_words]
                else:
                    tag_labels = [word.label for word in tagged_words]
                tag_idx = torch.stack([self.tag_tensors[self.tag_map.get(tag, UNK_ID)] for tag in tag_labels])
                tag_input = self.tag_embedding(tag_idx)
                word_inputs.append(tag_input)

            all_word_labels.append(word_labels)
            all_word_inputs.append(word_inputs)

        if forward_charlm is not None:
            all_forward_chars = self.build_char_representation(all_word_labels, device, forward_charlm, forward_charlm_vocab, forward=True)
            for word_inputs, forward_chars in zip(all_word_inputs, all_forward_chars):
                word_inputs.append(forward_chars)
        if backward_charlm is not None:
            all_backward_chars = self.build_char_representation(all_word_labels, device, backward_charlm, backward_charlm_vocab, forward=False)
            for word_inputs, backward_chars in zip(all_word_inputs, all_backward_chars):
                word_inputs.append(backward_chars)

        word_lstm_input = torch.zeros((max(len(x) for x in tagged_word_lists), len(tagged_word_lists), self.word_input_size), device=device)

        for sentence_idx, word_inputs in enumerate(all_word_inputs):
            # now of size sentence x input
            word_input = torch.cat(word_inputs, dim=1)
            word_input = self.word_dropout(word_input)

            word_lstm_input[:word_input.shape[0], sentence_idx, :] = word_input

        packed_word_input = torch.nn.utils.rnn.pack_padded_sequence(word_lstm_input, [len(x) for x in tagged_word_lists], enforce_sorted=False)
        word_output, _ = self.word_lstm(packed_word_input)
        # would like to do word_to_constituent here, but it seems PackedSequence doesn't support Linear
        # word_output will now be sentence x batch x 2*hidden_size
        word_output, word_output_lens = torch.nn.utils.rnn.pad_packed_sequence(word_output)
        # now sentence x batch x hidden_size

        word_queues = []
        for sentence_idx, tagged_words in enumerate(tagged_word_lists):
            sentence_output = word_output[:len(tagged_words), sentence_idx, :]
            sentence_output = self.word_to_constituent(sentence_output)
            sentence_output = self.nonlinearity(sentence_output)
            # TODO: this makes it so constituents downstream are
            # build with the outputs of the LSTM, not the word
            # embeddings themselves.  It is possible we want to
            # transform the word_input to hidden_size in some way
            # and use that instead
            word_queue = [WordNode(tag_node, sentence_output[idx, :])
                          for idx, tag_node in enumerate(tagged_words)]
            word_queue.append(WordNode(None, self.word_zeros))

            word_queues.append(word_queue)

        return word_queues
