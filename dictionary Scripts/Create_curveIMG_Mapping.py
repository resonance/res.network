import re
import os
import json


###############################

Mapping_file = "/Users/theodoreseem/Desktop/Emoji2Code/dictionaries/curveImage_mapping.json"
curveExtMap = "/Users/theodoreseem/Desktop/Emoji2Code/dictionaries/curveExtraction_mapping.json"
images_Directory = "/Users/theodoreseem/Desktop/Emoji2Code/Hasy-images/"
degrees = 360
distance = 200

###############################


def createMappingDictionary(cpts, images):
    outputFile = open(Mapping_file, "w")
    outputFile.write("{\n")
    imgNo = 4000
    for cp1 in cpts:
        for cp2 in cpts:
            mapping = "\"(%s, %s)\": \"%s\"," % (controlPointTypes[cp1],controlPointTypes[cp2], images[imgNo])
            outputFile.write(mapping + "\n")
            imgNo =  imgNo + 1
    outputFile.write("\"( , )\": \" \" \n")
    outputFile.write("}")


if __name__ == "__main__":

    with open(curveExtMap) as data_file:
        controlPointTypes = json.load(data_file)
    images = os.listdir(images_Directory)

    createMappingDictionary(controlPointTypes, images)
