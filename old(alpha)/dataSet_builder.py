import sys
import re
import os
import argparse
import random

Desired_Data_Size = 2;
StartEdit_Line = 13;
Measurement_File = "/Users/theodoreseem/Desktop/Emoji_Experiment/Tucker-Brianna.vit"
Seed_File = "/Users/theodoreseem/Desktop/Emoji_Experiment/Seed_Rectangle.val"
Output_Directory = "/Users/theodoreseem/Desktop/Emoji_Experiment/10-fold_images"

def storeMeasurement():
    measurList = []
    inputFile = open(Measurement_File, "r")
    for count, line in enumerate(inputFile):
        if "<m" in line:
            for word in line.split():
                if "name" in word:
                    ms = re.findall('"([^"]*)"', word)[0]
                    measurList.append(ms)
    print(measurList)
    return measurList

def createData(measurList):
    for number in range(1,Desired_Data_Size):
        outputFileName = "test"+str(number)+".val"
        print("Created file: ", outputFileName)
        createFile(outputFileName, measurList)
#(15,17),16

def createLength(measurList):
    randomInt = random.randint(1,10)
    substituteMeasure = measurList[randomInt]
    newWord = "length=\""+substituteMeasure+"\""
    return newWord

def createFile(outputFile, measurList):
    editableOutputFile = open(Output_Directory+outputFile, "w")
    readableInputFile = open(Seed_File, "r")

    wordOne = createLength(measurList)
    wordTwo = createLength(measurList)

    for counter, line in enumerate(readableInputFile):
        for char in line:
            if char == ' ': editableOutputFile.write(char)
            else: break;
        words = line.split()
        if counter==16:
            for word in words:
                if "length" in word:
                    editableOutputFile.write(wordOne+' ')
                else:
                    editableOutputFile.write(word+' ')
            editableOutputFile.write('\n')
        elif counter==15 or counter==17:
            for word in words:
                if "length" in word:
                    editableOutputFile.write(wordTwo+' ')
                else:
                    editableOutputFile.write(word+' ')
            editableOutputFile.write('\n')
        else:
            for word in words:
                editableOutputFile.write(word+' ')
            editableOutputFile.write('\n')
    editableOutputFile.close()
    readableInputFile.close()

if __name__ == "__main__":

        measurement_Dict = storeMeasurement()
        createData(measurement_Dict)
