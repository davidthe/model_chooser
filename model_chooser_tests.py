from app import model_select, images_path
from kraken.lib.xml import parse_alto
from os import walk
from evaluation import *
import Levenshtein


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


xmls_path = '/home/userm/Repositories/model_chooser/tests/xml_output/'
image_name = '348_3758c_default'
selected_models = {"ashkenazy": "/home/userm/Repositories/model_chooser/models/ashkenazy.mlmodel",
                   "sephardi": "/home/userm/Repositories/model_chooser/models/sephardi.mlmodel",
                   "vat44": "/home/userm/Repositories/model_chooser/models/vat44"
                            ".mlmodel"}
models_results = {}
# scores = model_select(images_path, selected_models, have_xml_outputs=True)
# just for now
scores = {'ashkenazy': 3998.738496383652, 'sephardi': 3887.967354422435, 'vat44': 3302.7467254278963}

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


for model in models_results.keys():
    # models_evaluations[model] = compare_triplet(' '.join(gt_lines), ' '.join(models_results[model]),
    #                                             ' '.join(models_results[model]))
    models_levenstian_distance[model] = calculate_levenshtein_distance(' '.join(gt_lines), ' '.join(models_results[model]))
    models_levenstian_distance_ratio[model] = calculate_levenshtein_distance(' '.join(gt_lines), ' '.join(models_results[model])) * 10

# create graph for each score

print(models_evaluations)
print(models_levenstian_distance)

import matplotlib.pyplot as plt
# Define the dictionaries
language_model_scores = scores
levenshtein_scores = models_levenstian_distance_ratio

# Get the keys from either dictionary (assuming they have the same keys)
evaluation_keys = list(language_model_scores.keys())

# Get the scores for each evaluation key
language_model_values = [language_model_scores[key] for key in evaluation_keys]
levenshtein_values = [levenshtein_scores[key] for key in evaluation_keys]

# Set the width of the bars
bar_width = 0.35

# Create an index for the x-axis
index = range(len(evaluation_keys))

# Create bar plots for language model and Levenshtein scores
plt.bar(index, language_model_values, bar_width, label='Language Model Scores')
plt.bar([i + bar_width for i in index], levenshtein_values, bar_width, label='Levenshtein Scores')

# Set x-axis labels
plt.xlabel('Evaluation Key')

# Set y-axis label
plt.ylabel('Scores')

# Set title
plt.title('Scores for Each Evaluation Key')

# Set x-axis ticks and labels
plt.xticks([i + bar_width/2 for i in index], evaluation_keys)

# Add a legend
plt.legend()

# Display the graph
plt.show()


