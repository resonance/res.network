import re
import os
import json


###############################

Mapping_file = "/Users/theodoreseem/res.Network/dictionaries/curve-Image_mapping.json"
Body_file = "/Users/theodoreseem/res.Network/Tucker-Brianna.vit"
GenMeasure_file = "/Users/theodoreseem/res.Network/dictionaries/lineExtraction_mapping.json"
images_Directory = "/Users/theodoreseem/res.Network/Hasy-images/"
degrees = 360

#Based on how many inputs in line-image mapping
startNum = 3000
###############################

def storeBodyMeasurement(measurList = []):
    inputFile = open(Body_file, "r")
    for count, line in enumerate(inputFile):
        if "<m" in line:
            msure = [re.findall('"([^"]*)"', word) for word in line.split() if "name" in word]
            measurList.append(msure[0].pop(0))
    return measurList

def storeGenericMeasurement(measurList = []):
    with open(GenMeasure_file) as data_file:
        measurements = json.load(data_file)
    measurements = [measurements[k] for k in measurements]
    return measurements


def storeDegrees(degList = []):
    for deg in range(0,degrees):
        degList.append(deg)
    return degList

def createMappingDictionary(measurements, degrees, images):
    outputFile = open(Mapping_file, "w")
    outputFile.write("{\n")
    lines = len(measurements)*len(degrees)
    print(len(measurements))
    imgNo = startNum
    for measure in measurements:
        for degree in range(0,len(degrees)):
            mapping = "\"(\'%s\', %d)\": \"%s\"," % (measure, degree, images[imgNo])
            outputFile.write(mapping + "\n")
            print(imgNo)
            imgNo = imgNo + 1
    outputFile.write("}")


if __name__ == "__main__":

    measurements =  storeGenericMeasurement()
    degrees = storeDegrees()
    images = os.listdir(images_Directory)

    createMappingDictionary(measurements, degrees, images)
