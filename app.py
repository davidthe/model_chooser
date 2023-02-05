import time
import warnings
from os import listdir
from os.path import isfile, join
from threading import RLock
from threading import Thread

import kraken.rpred
import mxnet as mx
from PIL import Image
from kraken import binarization
from kraken import blla
from kraken.lib import models
from kraken.lib import vgsl

from textScoreGenerator.mlm.src.mlm.models import get_pretrained
from textScoreGenerator.mlm.src.mlm.scorers import MLMScorerPT
from textScoreGenerator.tokenizer.dictatokenizer import DictaAutoTokenizer

warnings.filterwarnings("ignore")  # disable this line if u want to see all the warnings

model_lock = RLock()
printing_lock = RLock()

run_with_dicta_model = False

threads = []
images_threads = []
models_scores = {}
models_load_dict = {}
segmentations_dict = {}

ctxs = [mx.cpu()]  # or, e.g., [mx.gpu(0), mx.gpu(1)] todo try to use gpu
# ctxs = [mx.gpu(0), mx.gpu(1)]

segmentetion_model_path = 'segmentation_models/biblialong02_se3_2_tl.mlmodel'

vat44_image_path = 'pictures_examples/'

segment_model = vgsl.TorchVGSLModel.load_model(segmentetion_model_path)



if run_with_dicta_model:
    # init dicta model
    dicta_model_path = './textScoreGenerator/lm-dicta'
    dicta_tokenizer = DictaAutoTokenizer.from_pretrained(dicta_model_path)
    dicta_model, dicta_vocab, _ = get_pretrained(ctxs=ctxs, name="dicta", params_file=dicta_model_path)
    dicta_scorer = MLMScorerPT(dicta_model, dicta_vocab, dicta_tokenizer, ctxs)
else:
    model, vocab, tokenizer = get_pretrained(ctxs, 'onlplab/alephbert-base')
    scorer = MLMScorerPT(model, vocab, tokenizer, ctxs)

def get_image_text(model_name, baseline_seg, bw_im):
    model = models_load_dict[model_name]
    pred = kraken.rpred.rpred(model, bw_im, baseline_seg)

    return pred


def get_score_from_text(pred):
    # build text
    score = 0
    txt = ""
    for record in pred:
        txt += str(record) + ' '
        if len(txt) > 1:
            try:
                # with printing_lock:
                #     print(threading.get_native_id(), ": scoring txt: ", txt)
                #     print("score is: ", score)
                # score according to this https://github.com/awslabs/mlm-scoring
                if run_with_dicta_model:
                    score += (dicta_scorer.score_sentences([txt])[0] * -1)
                else:
                    score += (scorer.score_sentences([txt])[0] * -1)

                txt = ""
            except Exception as err:
                with printing_lock:
                    print("!!!!!!!!", err, "!!!!!!!!!")
                return score + 99999

    return score


def read_txt_and_score(baseline_seg, bw_im, model_name, image_name):
    if not (model_name in models_scores):
        with model_lock:
            models_scores[model_name] = 0

    start_time = time.time()

    with printing_lock:
        print("reading text of image:", image_name, " with model: ", model_name)

    # using kraken and ocr models
    pred = get_image_text(model_name, baseline_seg, bw_im)

    with printing_lock:
        print("--- reading one image took %s seconds ---" % (time.time() - start_time))

    start_time = time.time()

    # using language model
    with printing_lock:
        print("scoring: ", image_name, " with model: ", model_name)
    score = get_score_from_text(pred)

    with model_lock:
        models_scores[model_name] += score
        with printing_lock:
            print("current models scores \n", models_scores)

    with printing_lock:
        print("--- scoring one image took %s seconds ---" % (time.time() - start_time))


def finshed_threads_printer():
    should_run = True
    changed = -1
    start_time = time.time()

    while should_run:
        finshed_threads = 0

        for x in threads:
            if not x.is_alive():
                finshed_threads += 1

        if finshed_threads != changed:
            with printing_lock:
                print(finshed_threads, "threads finshed out of", len(threads))
            changed = finshed_threads

        if finshed_threads == len(threads) and len(threads) != 0:
            should_run = False
            print("--- %s seconds ---" % (time.time() - start_time))

        time.sleep(1)


def read_and_segment_image(imgs_path, image_name, segmentations):
    # Read the image via file.stream
    start_time = time.time()
    with printing_lock:
        print('starting file: ', join(imgs_path, image_name))
    img = Image.open(join(imgs_path, image_name))

    with printing_lock:
        print('convert image to black and white')
    # binarize image
    bw_im = binarization.nlbin(img)

    with printing_lock:
        print("--- making single image bw took %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    # segmentation
    if segmentations is None:
        baseline_seg = blla.segment(bw_im, model=segment_model)
    else:
        baseline_seg = segmentations[image_name]

    with printing_lock:
        print("--- segmenting single image took %s seconds ---" % (time.time() - start_time))

    segmentations_dict[image_name] = {"bw_im": bw_im, "baseline_seg": baseline_seg}

def model_select(imgs_path, models_dict, segmentations=None):
    '''
    :param imgs_path: str, Path to the folder containing the images to check
    :param models_dict: A dictionary of models: {"model_name1", "path_to_model", "model_name2", "path_to_model2"}
    :param optional segmentations: A dictionary of the images segmentations:
    {"imageName": segmentation object ( Dict[str, Any] )}
    :return models with accuracy :
    # rc = {"model1": "rank1", "model2": "rank2"}
    '''

    images = [f for f in listdir(imgs_path) if join(imgs_path, f).lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]

    t = Thread(target=finshed_threads_printer, args=[])
    t.start()

    # load all models so we dont need to load them any time
    start_time = time.time()
    with printing_lock:
        print('load all requested models')
    for model_name, path in models_dict.items():
        models_load_dict[model_name] = models.load_any(path)
    with printing_lock:
        print("--- loading models took %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    for image_name in images:
        t = Thread(target=read_and_segment_image, args=[imgs_path, image_name, segmentations])
        t.start()
        images_threads.append(t)

    for x in images_threads:
        x.join()

    with printing_lock:
        print("--- performing segmentation on all images took %s seconds ---" % (time.time() - start_time))

    for image_name in images:
        for model_name, _ in models_dict.items():
            with printing_lock:
                print("starting model: ", model_name)
            t = Thread(target=read_txt_and_score, args=[segmentations_dict[image_name]["baseline_seg"],
                                                        segmentations_dict[image_name]["bw_im"],
                                                        model_name, image_name])
            t.start()
            threads.append(t)

    # wait for all threads to finish
    for x in threads:
        x.join()
    # rc = {"model1": "rank1", "model2": "rank2"}
    return models_scores


# example
selected_models = {"ashkenazy": "models/ashkenazy.mlmodel", "sephardi": "models/sephardi.mlmodel",
                   "vat44": "models/vat44"
                            ".mlmodel"}

scores = model_select(vat44_image_path, selected_models)

with printing_lock:
    print(scores)
