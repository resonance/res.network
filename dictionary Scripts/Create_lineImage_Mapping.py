import re
import os


###############################

Mapping_file = "/Users/theodoreseem/Desktop/Emoji2Code/dictionaries/Line-Image_mapping.json"
Measurement_file = "/Users/theodoreseem/Desktop/Emoji2Code/Tucker-Brianna.vit"
images_Directory = "/Users/theodoreseem/Desktop/Emoji2Code/Hasy-images/"
degrees = 360

###############################

def storeMeasurement(measurList = []):
    inputFile = open(Measurement_file, "r")
    for count, line in enumerate(inputFile):
        if "<m" in line:
            msure = [re.findall('"([^"]*)"', word) for word in line.split() if "name" in word]
            measurList.append(msure[0].pop(0))
    return measurList

def storeDegrees(degList = []):
    for deg in range(0,degrees):
        degList.append(deg)
    return degList

def createMappingDictionary(measurements, degrees, images):
    outputFile = open(Mapping_file, "w")
    outputFile.write("{\n")
    lines = len(measurements)*len(degrees)
    print(len(measurements))
    imgNo = 0
    for measure in measurements:
        for degree in range(len(degrees)):
            mapping = "\"(\'%s\', %d)\": \"%s\"," % (measure, degree, images[imgNo])
            outputFile.write(mapping + "\n")
            print(imgNo)
            imgNo = imgNo + 1
    outputFile.write("}")


if __name__ == "__main__":

    measurements =  storeMeasurement()
    degrees = storeDegrees()
    images = os.listdir(images_Directory)

    createMappingDictionary(measurements[0:11], degrees, images)
