import os

import pytest

from stanza.models.constituency import parse_transitions
from stanza.models.constituency import tree_reader
from stanza.tests import *
from stanza.tests.constituency import test_parse_transitions
from stanza.tests.constituency.test_trainer import build_trainer, pt, TREEBANK

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

def build_model(pt, *args):
    trainer = build_trainer(pt, *args)
    return trainer.discriminator

def run_forward_tests(pt, *args):
    train_trees = tree_reader.read_trees(TREEBANK)
    discrim = build_model(pt, *args)
    result = discrim.forward(train_trees)
    assert len(result.shape) == 2
    assert result.shape[0] == 4
    assert result.shape[1] == 1

def test_forward(pt):
    run_forward_tests(pt)
