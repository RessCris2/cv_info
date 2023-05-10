import pycocotools
import pycococreatortools


INFO = {
    "description": "Example Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2018,
    "contributor": "waspinator",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]



CATEGORIES = [
    {
        'id': 1,
        'name': 'Epithelial',
        'supercategory': 'consep',
    },
    {
        'id': 2,
        'name': 'Inflammatory',
        'supercategory': 'consep',
    },
    {
        'id': 3,
        'name': 'Spindle-Shaped',
        'supercategory': 'consep',
    },
     {
        'id': 4,
        'name': 'Miscellaneous',
        'supercategory': 'consep',
    },
]



# loop through each image jpeg and its corresponding annotation pngs and let pycococreator generate the correctly formatted items. 