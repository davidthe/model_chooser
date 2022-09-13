import kraken.rpred
from flask import Flask, redirect, request, jsonify
from PIL import Image
from datetime import datetime
from kraken import binarization
from kraken import pageseg
from kraken import blla
from kraken.lib import vgsl
from kraken.lib import models
from kraken import serialization
from kraken import rpred

segmentetion_model_path = 'models/biblialong02_se3_2_tl.mlmodel'
recognition_model_path = 'models/sephardi_01.mlmodel'
segment_model = vgsl.TorchVGSLModel.load_model(segmentetion_model_path)
rec_model = models.load_any(recognition_model_path)
# img = Image.open("none1.png")
# baseline_seg = blla.segment(img, model=segment_model)
# print(baseline_seg)
app = Flask(__name__)


@app.route('/')
def homepage():
    return redirect("https://sofer.info/", code=302)


@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files['image']
        # Read the image via file.stream
        img = Image.open(file.stream)
        # binarize image
        bw_im = binarization.nlbin(img)
        # segmentation
        baseline_seg = blla.segment(bw_im, model=segment_model)
        # recogine text
        pred_it = kraken.rpred.rpred(rec_model, bw_im, baseline_seg)

        # build response
        final_text = ''
        for record in pred_it:
            print(record)
            final_text += str(record)

        return jsonify({'msg': 'success', 'size': [img.width, img.height], 'text': final_text})
    except Exception as err:
        print(err)


if __name__ == '__main__':
    app.run()


def get_image_text(path_to_model, baseline_seg, bw_im):
    model = models.load_any(path_to_model)
    pred = kraken.rpred.rpred(model, bw_im, baseline_seg)

    # build text
    final_text = ''
    for record in pred:
        final_text += str(record)

    return final_text


def get_score_from_text(text):
    return 98.2


def model_select(img_path, models_dict):
    '''
    :param img_path: str, Path to an image to check
    :param models_dict: A dictionary of models: {"model_name1", "path_to_model", "model_name2", "path_to_model2"}
    :return models with accuracy :
    '''
    # Read the image via file.stream
    img = Image.open(img_path)
    # binarize image
    bw_im = binarization.nlbin(img)
    # segmentation
    baseline_seg = blla.segment(bw_im, model=segment_model)

    models_scores = {}

    for name, path in models_dict.items():
        # using kraken and ocr models
        text = get_image_text(path, baseline_seg, bw_im)
        # using language model
        score = get_score_from_text(text)
        models_scores[name] = score

    # Check image format

    # Check models paths

    # rc = {"model1": "rank1", "model2": "rank2"}
    return models_scores


