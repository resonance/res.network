import re
import os
import json


###############################
## TODO: After running this script, NEED TO REMOVE COMMA FROM LAST LINE OF CREATED DICTIONARY
## TODO: Change For line and curve mapping; 1) Change name to Line-Image or Curve-Image; 2) Change starting image number



# )1
Mapping_file = "/Users/theodoreseem/res.Network/Curve_Integration/dictionaries/Curve-Image.json"

Body_file = "/Users/theodoreseem/res.Network/Tucker-Brianna.vit"
GenMeasure_file = "/Users/theodoreseem/res.Network/Curve_Integration/dictionaries/Feature-Extraction.json"
images_Directory = "/Users/theodoreseem/res.Network/Hasy-images/"
degrees = 360

# )2
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
    imgNo = startNum
    for measure in measurements:
        for degree in range(0,len(degrees),10):
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
