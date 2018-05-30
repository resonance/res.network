import re
import os


###############################

Mapping_file = "/Users/theodoreseem/res.Network/dictionaries/curveExtraction_mapping.json"
images_Directory = "/Users/theodoreseem/res.Network/CIFAR10-images/"
degrees = 360
distance = 1000

###############################


def storeDegrees(degList = []):
    for deg in range(0,degrees,10):
        degList.append(deg)
    return degList

def storeDistance(disList = []):
    for dis in range(0,distance+1,20):
        disList.append(dis)
    return disList

def createMappingDictionary(degrees, distances):
    outputFile = open(Mapping_file, "w")
    outputFile.write("{\n")
    cpNum = 0
    controlPointType = "ControlPointType-" + str(cpNum)
    for distance in distances:
        for degree in degrees:
            mapping = "\"(\'%d\', %d)\": \"%s\"," % (distance, degree, controlPointType)
            outputFile.write(mapping + "\n")
            cpNum =  cpNum + 1
            controlPointType = "ControlPointType-" + str(cpNum)
    outputFile.write("\"(\'%d\', %d)\": \"%s\"" % (distances[-1], degrees[-1], controlPointType) + "\n")
    outputFile.write("}")



if __name__ == "__main__":

    degrees = storeDegrees()
    distances = storeDistance()


    createMappingDictionary(degrees, distances)
