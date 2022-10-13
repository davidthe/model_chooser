import kraken.rpred
import mxnet as mx
from PIL import Image
from kraken import binarization
from kraken import blla
from kraken.lib import models
from kraken.lib import vgsl
from os import listdir
from os.path import isfile, join
from threading import RLock
from threading import Thread
import time

lock = RLock()
threads = []
models_scores = {}

from textScoreGenerator.mlm.src.mlm.models import get_pretrained
from textScoreGenerator.mlm.src.mlm.scorers import MLMScorerPT

ctxs = [mx.cpu()]  # or, e.g., [mx.gpu(0), mx.gpu(1)]

segmentetion_model_path = 'segmentation_models/biblialong02_se3_2_tl.mlmodel'

vat44_image_path = 'pictures_examples/'

segment_model = vgsl.TorchVGSLModel.load_model(segmentetion_model_path)

model, vocab, tokenizer = get_pretrained(ctxs, 'onlplab/alephbert-base')  # todo make dicta model support this
scorer = MLMScorerPT(model, vocab, tokenizer, ctxs)


def get_image_text(path_to_model, baseline_seg, bw_im):
    model = models.load_any(path_to_model)
    pred = kraken.rpred.rpred(model, bw_im, baseline_seg)

    return pred


def get_score_from_text(pred):
    # build text
    score = 0
    for record in pred:
        txt = str(record)
        if len(txt) > 0:
            score += scorer.score_sentences([txt])[0]

    return score


def model_select(imgs_path, models_dict):
    '''
    :param imgs_path: str, Path to an image to check
    :param models_dict: A dictionary of models: {"model_name1", "path_to_model", "model_name2", "path_to_model2"}
    :return models with accuracy :
    '''

    onlyfiles = [f for f in listdir(imgs_path) if isfile(join(imgs_path, f))]


    t = Thread(target=finshed_threads_printer, args=[])
    t.start()

    for file in onlyfiles:
        # Read the image via file.stream
        print('starting file: ', join(imgs_path, file))
        img = Image.open(join(imgs_path, file))
        # binarize image
        bw_im = binarization.nlbin(img)
        # segmentation
        baseline_seg = blla.segment(bw_im, model=segment_model)


        for name, path in models_dict.items():
            print("starting model: ", name)
            t = Thread(target=read_txt_and_score, args=[baseline_seg, bw_im, name, path])
            t.start()
            threads.append(t)

    # Check image format

    # wait for all threads to finish
    for x in threads:
        x.join()
    # rc = {"model1": "rank1", "model2": "rank2"}
    return models_scores


def read_txt_and_score(baseline_seg, bw_im, name, path):
    if not (name in models_scores):
        with lock:
            models_scores[name] = 0
    # using kraken and ocr models
    pred = get_image_text(path, baseline_seg, bw_im)
    # using language model
    score = get_score_from_text(pred)
    with lock:
        models_scores[name] += score

def finshed_threads_printer():
    should_run = True

    while should_run:
        finshed_threads = 0
        for x in threads:
            if not x.is_alive():
                finshed_threads += 1

        print(finshed_threads, " finshed out of ", len(threads))

        if finshed_threads == len(threads) and len(threads) != 0:
            should_run = False

        time.sleep(1)


selected_models = {"ashkenazy": "models/ashkenazy.mlmodel", "sephardi": "models/sephardi.mlmodel", "vat44": "models/vat44"
                                                                                                   ".mlmodel"}
scores = model_select(vat44_image_path, selected_models)

print(scores)
