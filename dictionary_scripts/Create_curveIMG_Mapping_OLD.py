import re
import os
import json


###############################

Mapping_file = "/Users/theodoreseem/res.Network/Curve_Integration/dictionaries/Curve-Image.json"
curveExtMap = "/Users/theodoreseem/res.Network/Curve_Integration/dictionaries/Curve-Extraction.json"
images_Directory = "/Users/theodoreseem/res.Network/Hasy-images/"

#Based on how many innputs in line-image mapping
startNum = 3000
###############################


def createMappingDictionary(cpts, images):
    outputFile = open(Mapping_file, "w")
    outputFile.write("{\n")
    imgNum = startNum
    for cp in cpts:
            mapping = "\"(%s)\": \"%s\"," % (controlPointTypes[cp], images[imgNum])
            outputFile.write(mapping + "\n")
            imgNum =  imgNum + 1
    outputFile.write("\"( , )\": \" \" \n")
    outputFile.write("}")


if __name__ == "__main__":

    with open(curveExtMap) as data_file:
        controlPointTypes = json.load(data_file)
    images = os.listdir(images_Directory)

    createMappingDictionary(controlPointTypes, images)
