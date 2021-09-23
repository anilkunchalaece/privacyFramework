"""
Author : Kunchala Anil
Email : D20125529@mytudublin.ie

This script is used to extract the data related to the attrbutes of intrest.
Currently we are intrested in following attrbutes

                                RAP Attributes
1. Gender                   'Female'
2. Age                      'AgeLess16','Age17-30','Age31-45','Age46-60'
3. Hair Style               'hs-BaldHead','hs-LongHair','hs-BlackHair','hs-Hat','hs-Glasses'
4. Body Type                'BodyFat', 'BodyNormal', 'BodyThin'
4. Upper clothing info      'ub-Shirt', 'ub-Sweater', 'ub-Vest', 'ub-TShirt', 'ub-Cotton', 'ub-Jacket', 'ub-SuitUp', 'ub-Tight', 'ub-ShortSleeve', 'ub-Others'
5. Lower clothing info      'lb-LongTrousers', 'lb-Skirt', 'lb-ShortSkirt', 'lb-Dress', 'lb-Jeans', 'lb-TightTrousers'
6. Accessories              'attachment-Backpack', 'attachment-ShoulderBag', 'attachment-HandBag', 'attachment-Box', 'attachment-PlasticBag', 'attachment-PaperBag', 'attachment-HandTrunk', 'attachment-Other',
"""
import numpy as np
import os
import pickle

ATTR_OF_INTREST = [
    'hs-BaldHead','hs-LongHair','hs-BlackHair','hs-Hat','hs-Glasses',
    'ub-Shirt', 'ub-Sweater', 'ub-Vest', 'ub-TShirt', 'ub-Cotton', 'ub-Jacket', 'ub-SuitUp', 'ub-Tight', 'ub-ShortSleeve', 'ub-Others',
    'lb-LongTrousers', 'lb-Skirt', 'lb-ShortSkirt', 'lb-Dress', 'lb-Jeans', 'lb-TightTrousers',
    'attachment-Backpack', 'attachment-ShoulderBag', 'attachment-HandBag', 'attachment-Box', 'attachment-PlasticBag', 'attachment-PaperBag', 'attachment-HandTrunk', 'attachment-Other',
    'AgeLess16','Age17-30','Age31-45','Age46-60',
    'Femal',
    'BodyFat', 'BodyNormal', 'BodyThin',
]



def loadDataFromFile(fileName) :
    # fileName -> location of dataset.pkl file
    print(F"loading data from file {fileName}")
    with open(fileName,'rb') as f :
        data = pickle.load(f)
    return data

def processData(srcDir):
    fileName = os.path.join(srcDir,'dataset.pkl')
    data = loadDataFromFile(fileName)
    attr = data['attr_name']

    selectedAttributeIndex = []
    # extract index for selected attributes
    print(F"Total number of attributes in original dataset are {len(attr)}")
    for a in ATTR_OF_INTREST :
        idx = attr.index(a)
        selectedAttributeIndex.append(idx)
    print(F"Number of selected attribuates are {len(selectedAttributeIndex)}")
    selectedAttributeIndex = np.array(selectedAttributeIndex)
    attr = np.array(attr)

    labelNames = attr[selectedAttributeIndex]

    orgLabels = data["label"]
    print(F"Original dataset label size is{orgLabels.shape}")

    labels = data["label"][:,selectedAttributeIndex]
    print(F"Modified dataset labels has size {labels.shape}")
    
    # add directory info to the image path
    imageNames = [os.path.join(srcDir,"RAP_dataset",name) for name in data["image_name"]]

    finalList = [(imageNames[i],labels[i]) for i in range(len(imageNames))]

    return {
        "dataset" : finalList,
        "label_names" : labelNames,
        # "labels" : labels
    }



if __name__ == "__main__" :
    srcDir = "/home/akunchala/Documents/z_Datasets/RAP"
    data = processData(srcDir)
    print(len(data["dataset"]))
    print(data.keys())