"""The DCC speed-ups give the tables of the upstream code paths, independent of batch composition."""

import copy
import gc
import json
import os
import weakref

import cv2
import pytest
import torch

from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor
from docling_ibm_models.tableformer.dcc_flags import int_env
from docling_ibm_models.tableformer.models.table04_rs.tag_decoder_rs import TagDecoders

from tests.test_tf_predictor import docling_api_data, init, test_config  # noqa: F401

SWITCHES = ("MATCH", "DECODER", "PREP")


def _pages():
    pages = []
    for table_json_fn, png_image_fn, table_bboxes in zip(
        docling_api_data["table_jsons"], docling_api_data["png_images"], docling_api_data["table_bboxes"]
    ):
        with open(table_json_fn, "r") as fp:
            page = json.load(fp)["pages"][0]
        page["image"] = cv2.imread(png_image_fn)
        page["table_bboxes"] = table_bboxes
        pages.append(page)
    return pages


def _run(config, monkeypatch, value: str):
    for name in SWITCHES:
        monkeypatch.setenv(f"DCC_TABLES_{name}", value)
    predictor = TFPredictor(copy.deepcopy(config), device="cpu", num_threads=2)
    out = []
    for page in _pages():
        for matching in (True, False):
            out.append(predictor.multi_table_predict(page, page["table_bboxes"], do_matching=matching))
    return repr(out)


def test_switches_give_the_upstream_tables(init, monkeypatch):  # noqa: F811
    """All switches off (upstream code paths) and on give the same predictions, matched and unmatched."""
    config = init[0]
    assert _run(config, monkeypatch, "0") == _run(config, monkeypatch, "1")


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_a_table_decodes_the_same_whatever_shares_its_batch(init, device):  # noqa: F811
    """Decoding pads every batch to the same size, so a table's result does not depend on its neighbours."""
    predictor = TFPredictor(copy.deepcopy(init[0]), device=device, num_threads=2)
    model = predictor._model
    torch.manual_seed(0)
    imgs = torch.rand(3, 3, 448, 448, device=device)
    alone = model.predict_batch(imgs[:1], 1024)[0]
    shared = model.predict_batch(imgs, 1024)[0]
    assert alone[0] == shared[0]
    assert torch.equal(alone[1], shared[1]) and torch.equal(alone[2], shared[2])


def test_decoders_hold_their_model_weakly():
    class Model:
        pass

    model = Model()
    decoders = TagDecoders(model, max_steps=8)
    ref = weakref.ref(model)
    del model
    gc.collect()
    assert ref() is None
    with pytest.raises(RuntimeError):
        decoders.get()


@pytest.mark.parametrize(("value", "expected"), [(None, 4), ("", 4), ("x", 4), ("0", 1), ("-3", 1), ("2", 2), ("8", 8)])
def test_int_env(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("DCC_TABLES_MAXB", raising=False)
    else:
        monkeypatch.setenv("DCC_TABLES_MAXB", value)
    assert int_env("MAXB", 4) == expected
