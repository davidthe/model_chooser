import time
import warnings
import os
import copy
from pathlib import Path
from os import listdir
from os.path import join
from threading import RLock
from threading import Thread

import kraken.rpred
import mxnet as mx
import torch
from PIL import Image
from kraken import binarization
from kraken import blla
from kraken.lib import models
from kraken.lib import vgsl
from kraken import serialization

from textScoreGenerator.mlm.src.mlm.models import get_pretrained
from textScoreGenerator.mlm.src.mlm.scorers import MLMScorerPT
from textScoreGenerator.tokenizer.dictatokenizer import DictaAutoTokenizer


REPO_ROOT = Path(__file__).resolve().parent
SEGMENTATION_MODEL_PATH = REPO_ROOT / "segmentation_models" / "biblialong02_se3_2_tl.mlmodel"
RECOGNITION_MODELS_PATH = REPO_ROOT / "recognition_models"
DEFAULT_IMAGES_PATH = REPO_ROOT / "pictures_examples"
DEFAULT_XML_OUTPUT_PATH = REPO_ROOT / "xml_output"
DICTA_MODEL_PATH = REPO_ROOT / "textScoreGenerator" / "lm-dicta"
TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MX_CONTEXTS = [mx.gpu(0)] if mx.context.num_gpus() > 0 else [mx.cpu()]


class django_setting():
    def __init__(self):
        self.PIPELINE = {"segmentation_model": str(SEGMENTATION_MODEL_PATH),
                         "out_trans_path": str(DEFAULT_XML_OUTPUT_PATH)}

try:
    from django.conf import settings
except Exception:
    settings = django_setting()

# remove this
settings = django_setting()

warnings.filterwarnings("ignore")  # disable this line if u want to see all the warnings

model_lock = RLock()
printing_lock = RLock()
xml_outputs = False
run_with_dicta_model = True
number_of_lines_to_concat = 1

threads = []
images_threads = []
models_scores = {}
models_load_dict = {}
segmentations_dict = {}
xml_dict = {}

ctxs = MX_CONTEXTS

# segmentetion_model_path = 'segmentation_models/biblialong02_se3_2_tl.mlmodel'
segmentetion_model_path = settings.PIPELINE["segmentation_model"]
xml_output_path = settings.PIPELINE["out_trans_path"]
images_path = str(DEFAULT_IMAGES_PATH)

segment_model = vgsl.TorchVGSLModel.load_model(segmentetion_model_path)

if run_with_dicta_model:
    # init dicta model
    dicta_model_path = str(DICTA_MODEL_PATH)
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
    lines = [str(record) for record in pred]
    if not lines:
        return 0

    chunk_size = max(1, int(number_of_lines_to_concat))
    score = 0

    for start in range(0, len(lines), chunk_size):
        txt = "\n".join(lines[start:start + chunk_size]) + "\n"
        try:
            # score according to this https://github.com/awslabs/mlm-scoring
            if run_with_dicta_model:
                score += (dicta_scorer.score_sentences([txt])[0] * -1)
            else:
                score += (scorer.score_sentences([txt])[0] * -1)
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
    new_pred = copy.deepcopy(pred)

    with printing_lock:
        print("--- reading one image took %s seconds ---" % (time.time() - start_time))

    start_time = time.time()

    # using language model
    with printing_lock:
        print("scoring: ", image_name, " with model: ", model_name)
    score = get_score_from_text(pred)
    with printing_lock:
        print(score)

    with model_lock:
        models_scores[model_name] += score
        with printing_lock:
            print("current models scores \n", models_scores)

    with printing_lock:
        print("--- scoring one image took %s seconds ---" % (time.time() - start_time))

    # build alto from ocr response
    recs = [r for r in new_pred]
    with printing_lock:
        print(recs)
    alto = serialization.serialize(recs, image_name=image_name, image_size=bw_im.size,
                                   template='alto')
    xml_dict[f"{model_name}__{Path(image_name).stem}"] = alto
    return score


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


def read_and_segment_image(imgs_path, image_name, segmentations, device=None):
    # Read the image via file.stream
    start_time = time.time()
    with printing_lock:
        print('starting file: ', join(imgs_path, image_name))
    img = Image.open(join(imgs_path, image_name))

    with printing_lock:
        print('convert image to black and white')
    # binarize image
    # bw_im = binarization.nlbin(img)
    bw_im = img

    with printing_lock:
        print("--- making single image bw took %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    # segmentation
    if segmentations is None:
        baseline_seg = blla.segment(bw_im, model=segment_model, device=device or TORCH_DEVICE)
    else:
        baseline_seg = segmentations[image_name]

    with printing_lock:
        print("--- segmenting single image took %s seconds ---" % (time.time() - start_time))

    segmentations_dict[image_name] = {"bw_im": bw_im, "baseline_seg": baseline_seg}


def model_select(imgs_path, models_dict, segmentations=None, have_xml_outputs=False, concat_lines = 1):
    '''
    :param have_xml_outputs define if xmls of all the models will be saved
    :param imgs_path: str, Path to the folder containing the images to check
    :param models_dict: A dictionary of models: {"model_name1", "path_to_model", "model_name2", "path_to_model2"}
    :param optional segmentations: A dictionary of the images segmentations:
    {"imageName": segmentation object ( Dict[str, Any] )}
    :return models with accuracy :
    # rc = {"model1": "rank1", "model2": "rank2"}
    '''
    global number_of_lines_to_concat, xml_outputs, threads, images_threads, models_scores, models_load_dict, segmentations_dict, xml_dict

    number_of_lines_to_concat = concat_lines
    xml_outputs = have_xml_outputs
    threads = []
    images_threads = []
    models_scores = {}
    models_load_dict = {}
    segmentations_dict = {}
    xml_dict = {}
    images = sorted([f for f in listdir(imgs_path) if
              join(imgs_path, f).lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif'))])

    t = Thread(target=finshed_threads_printer, args=[])
    t.start()

    # load all models so we dont need to load them any time
    start_time = time.time()
    with printing_lock:
        print('load all requested models')
    for model_name, path in models_dict.items():
        models_load_dict[model_name] = models.load_any(path, device=TORCH_DEVICE)
    with printing_lock:
        print("--- loading models took %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    for image_name in images:
        t = Thread(target=read_and_segment_image, args=[imgs_path, image_name, segmentations, TORCH_DEVICE])
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

    # ---- write xml files --------------
    try:
        selected_model = min(models_scores, key=models_scores.get)

        if not os.path.isdir(xml_output_path):
            os.makedirs(xml_output_path)
        if xml_outputs:
            keys_xmls = [k for k in list(xml_dict.keys())]
        else:
            keys_xmls = [k for k in list(xml_dict.keys()) if selected_model in k]

        for key in keys_xmls:
            with open(Path(xml_output_path) / f"{key}.xml", 'w') as fp:
                fp.write(xml_dict[key])
    except Exception:
        print("fail to save output xmls")

    # -----------------------------------

    # rc = {"model1": "rank1", "model2": "rank2"}
    return models_scores


# example


def discover_recognition_models(models_root=RECOGNITION_MODELS_PATH):
    models_root = Path(models_root)
    return {path.stem: str(path) for path in sorted(models_root.glob("*.mlmodel"))}


selected_models = discover_recognition_models()

# scores = model_select(images_path, selected_models, have_xml_outputs=True, concat_lines=2)
#
# with printing_lock:
#     print(scores)
