# from app import model_select, images_path
from kraken.lib.xml import parse_alto
from os import walk
from evaluation import *
import Levenshtein
from jiwer import wer, cer


def calculate_levenshtein_distance(gt_text, ocr_text):
    """
    Calculate the Levenshtein distance between two strings.

    Parameters:
    - gt_text: The ground truth text.
    - ocr_text: The OCR-generated text.

    Returns:
    - The Levenshtein distance between the two strings.
    """
    return Levenshtein.distance(gt_text, ocr_text)


def calculate_wer(gt_text, ocr_text):
    return wer(gt_text, ocr_text)


def calculate_cer(gt_text, ocr_text):
    return cer(gt_text, ocr_text)


xmls_path = '/home/userm/Repositories/model_chooser/xml_output/'
image_name = '348_3758c_default'
selected_models = {"ashkenazy": "/home/userm/Repositories/model_chooser/models/ashkenazy.mlmodel",
                   "sephardi": "/home/userm/Repositories/model_chooser/models/sephardi.mlmodel",
                   "vat44": "/home/userm/Repositories/model_chooser/models/vat44.mlmodel",
                   "bibilia9": "/home/userm/Repositories/model_chooser/models/biblia9.mlmodel"}
images_path = '/home/userm/Repositories/model_chooser/pictures_examples/'
models_results = {}
# scores = model_select(images_path, selected_models, have_xml_outputs=True)
# just for now
scores = {'ashkenazy': 3219.6986465500668, 'sephardi': 3097.8383238322567, 'vat44': 2392.2141431979635, 'bibilia9': 1979.1274481718428}
print(scores)


# load gt from gt folder
with open(images_path + 'gt_for_pics/348_3758c_default.txt') as f:
    gt_lines = f.readlines()
    print(len(gt_lines))
    print(gt_lines)

# load xmls outputs of moddels
filenames = next(walk(xmls_path), (None, None, []))[2]  # [] if no file
for file in filenames:
    val = parse_alto(xmls_path + file)
    line_texts = [line['text'] + '\n' for line in val['lines']]
    print(len(line_texts))
    print(line_texts)
    models_results[file.split(image_name)[0]] = line_texts


# calculate wer cer and do levenstian ocr-output -> gt
models_evaluations = {}
models_levenstian_distance = {}
models_levenstian_distance_ratio = {}
models_wer_scores = {}
models_cer_scores = {}


for model in models_results.keys():
    # models_evaluations[model] = compare_triplet(' '.join(gt_lines), ' '.join(models_results[model]),
    #                                             ' '.join(models_results[model]))
    gt = ' '.join(gt_lines)
    model_response = ' '.join(models_results[model])
    print("model: ", model, " response: ", model_response)
    models_levenstian_distance[model] = calculate_levenshtein_distance(gt, model_response)
    models_levenstian_distance_ratio[model] = calculate_levenshtein_distance(gt, model_response) * 10
    models_wer_scores[model] = calculate_wer(gt, model_response) * 100
    models_cer_scores[model] = calculate_cer(gt, model_response) * 100
# create graph for each score

print(models_evaluations)
print('levenstain: ',models_levenstian_distance)
print('wer: ',models_wer_scores)
print('cer: ',models_cer_scores)

import matplotlib.pyplot as plt
# Define the dictionaries
language_model_scores = scores
levenshtein_scores = models_levenstian_distance
wer_scores = models_wer_scores
cer_scores = models_cer_scores

# Get the keys from either dictionary (assuming they have the same keys)
evaluation_keys = list(language_model_scores.keys())

# Get the scores for each evaluation key
language_model_values = [language_model_scores[key] / 10 for key in evaluation_keys]
levenshtein_values = [levenshtein_scores[key] for key in evaluation_keys]
cer_values = [cer_scores[key] for key in evaluation_keys]
wer_values = [wer_scores[key] for key in evaluation_keys]
# Set the width of the bars
bar_width = 0.20

# Create an index for the x-axis
index = range(len(evaluation_keys))

# Create bar plots for language model and Levenshtein scores
plt.bar([i - bar_width for i in index], language_model_values, bar_width, label='Language Model Scores')
plt.bar([i + bar_width- bar_width for i in index], levenshtein_values, bar_width, label='Levenshtein Scores')
plt.bar([i + (bar_width*2)- bar_width for i in index], wer_values, bar_width, label='WER Scores (percentage)')
plt.bar([i + (bar_width*3)- bar_width for i in index], cer_values, bar_width, label='CER Scores (percentage)')

# Set x-axis labels
plt.xlabel('HTR Models')

# Set y-axis label
plt.ylabel('Scores')

# Set title
plt.title('Scores for Each HTR model')

# Set x-axis ticks and labels
plt.xticks([i + bar_width/2 for i in index], evaluation_keys)

# Add a legend
plt.legend()

# Display the graph
plt.show()


