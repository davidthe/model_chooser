from app import model_select
from kraken.lib.xml import parse_xml

images_path = 'pictures_examples/'
xmls_path = 'xml_output/'


selected_models = {"ashkenazy": "/home/userm/Repositories/model_chooser/models/ashkenazy.mlmodel", "sephardi": "/home/userm/Repositories/model_chooser/models/sephardi.mlmodel",
                   "vat44": "/home/userm/Repositories/model_chooser/models/vat44"
                            ".mlmodel"}

scores = model_select(images_path, selected_models, have_xml_outputs=True)

# load gt from gt folder
with open(images_path + 'gt_for_pics/348_3758c_default.txt') as f:
    gt_lines = f.readlines()

# load xmls outputs of moddels
val = parse_xml(xmls_path+'348_3758c_default.xml')
# calculate wer cer and do levenstian ocr-output -> gt

# create graph for each score

