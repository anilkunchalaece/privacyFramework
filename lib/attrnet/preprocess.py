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
from sklearn.model_selection import train_test_split

ATTR_OF_INTREST = [
    'hs-BaldHead','hs-LongHair','hs-BlackHair','hs-Hat','hs-Glasses',
    'ub-Shirt', 'ub-Sweater', 'ub-Vest', 'ub-TShirt', 'ub-Cotton', 'ub-Jacket', 'ub-SuitUp', 'ub-Tight', 'ub-ShortSleeve', 'ub-Others',
    'lb-LongTrousers', 'lb-Skirt', 'lb-ShortSkirt', 'lb-Dress', 'lb-Jeans', 'lb-TightTrousers',
    'attachment-Backpack', 'attachment-ShoulderBag', 'attachment-HandBag', 'attachment-Box', 'attachment-PlasticBag', 'attachment-PaperBag', 'attachment-HandTrunk', 'attachment-Other',
    'AgeLess16','Age17-30','Age31-45','Age46-60',
    'Femal',
    'BodyFat', 'BodyNormal', 'BodyThin',
]

ATTR_OF_INTREST_IDX=[
    [0,1,2,3,4],
    [5,6,7,8,9,10,11,12,13,14],
    [15,16,17,18,19,20],
    [21,22,23,24,25,26,27,28],
    [29,30,31,32],
    [33],
    [34,35,36]
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

    # label "female" has value 2 in very rare cases, replace with 1
    # labels = np.delete(labels,np.where(labels == 2))
    labels = np.where(labels == 2,1, labels)
    
    # add directory info to the image path
    imageNames = [os.path.join(srcDir,"RAP_dataset",name) for name in data["image_name"]]

    finalList = [(imageNames[i],labels[i]) for i in range(len(imageNames))]

    testList = [(imageNames[idx] , labels[idx]) for idx in data["partition"]["test"][0]]
    trainList = [(imageNames[idx] , labels[idx]) for idx in data["partition"]["train"][0]]
    validList = [(imageNames[idx] , labels[idx]) for idx in data["partition"]["val"][0]]


    return {
        # "dataset" : finalList,
        "test" : testList,
        "train" : trainList,
        "valid" : validList,
        "label_names" : labelNames,
        # "labels" : labels
    }

# we dont need to split the dataset specifically
# use the parition given in the RAP dataset
def splitData(srcDir):
    # split data into train, test and valid
    _data = processData(srcDir)
    # allData = _data["dataset"]
    # train , test_valid = train_test_split(allData,test_size=0.30)
    # test , valid = train_test_split(test_valid,test_size=0.50)
    # print(F"Total data {len(allData)} , train {len(train)}, test {len(test)}, valid {len(valid)}")
    # dataset = {
    #     "train" : train,
    #     "test" : test,
    #     "valid" : valid,
    #     "label_names" : _data["label_names"]
    # }
    outFileName = os.path.join(srcDir,"dataset_modified.pkl")
    with open(outFileName,"wb") as fd: 
        pickle.dump(_data,fd)

if __name__ == "__main__" :
    srcDir = "/home/akunchala/Documents/z_Datasets/RAP"
    data = processData(srcDir)
    splitData(srcDir)
    _d = loadDataFromFile(os.path.join(srcDir,"dataset_modified.pkl"))
    print(_d.keys())
