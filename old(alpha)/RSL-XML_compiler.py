#!/usr/bin/env python
from __future__ import print_function

import sys
import os
import re

dsl_path = "assets/rsl-seamly-mapping.json"

InputDirectory = "/Users/theodoreseem/Desktop/Emoji_Experiment/test_compile/"
OutputDirectory = "/Users/theodoreseem/Desktop/Emoji_Experiment/test_compile/"

repeatLines = ["<?xml version=\"1.0\" encoding=\"UTF-8\"?>", "<pattern>", "<!--Pattern created with Seamly2D v0.6.0.0a (https://fashionfreedom.eu/).-->", "<version>0.6.0</version>",
                "<unit>inch</unit>", "<description/>","<notes/>","<measurements>/Users/theodoreseem/ResonanceHub/image2XML/image2XML/Data/Vector_Measurements/Tucker-Brianna.vit</measurements>",
                "<increments/>", "<draw name=\"Test1\">", "<calculation>", "<point id=\"1\" mx=\".1\" my=\".1\" name=\"A\" type=\"single\" x=\"90\" y=\"90\"/>",
                "<point angle=\"90\" basePoint=\"1\" id=\"2\" length=\"height*2\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"A1\" type=\"endLine\" typeLine=\"hair\"/>",
                "<point angle=\"180\" basePoint=\"1\" id=\"3\" length=\"height*2\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"A2\" type=\"endLine\" typeLine=\"hair\"/>",
                "<point id=\"4\" mx=\".1\" my=\".1\" name=\"B\" type=\"single\" x=\"20\" y=\"20\"/>"]

repeatLines2 = ["</calculation>","<modeling/>","<details/>","</draw>","</pattern>"]

def RSLSub(x):
    return {
          "neck-back-0": "<point angle=\"360\" basePoint=\"5\" id=\"6\" length=\"height_neck_back\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"B2\" type=\"endLine\" typeLine=\"hair\"/>",
          "neck-back-90": "<point angle=\"90\" basePoint=\"6\" id=\"7\" length=\"height_neck_back\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"B3\" type=\"endLine\" typeLine=\"hair\"/>",
          "neck-back-270":"<point angle=\"270\" basePoint=\"4\" id=\"5\" length=\"height_neck_back\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"B1\" type=\"endLine\" typeLine=\"hair\"/>",
          "scapula-0": "<point angle=\"360\" basePoint=\"5\" id=\"6\" length=\"height_scapula\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"B2\" type=\"endLine\" typeLine=\"hair\"/>",
          "scapula-90": "<point angle=\"90\" basePoint=\"6\" id=\"7\" length=\"height_scapula\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"B3\" type=\"endLine\" typeLine=\"hair\"/>",
          "scapula-270": "<point angle=\"270\" basePoint=\"4\" id=\"5\" length=\"height_scapula\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"B1\" type=\"endLine\" typeLine=\"hair\"/>",
          "armpit-0": "<point angle=\"360\" basePoint=\"5\" id=\"6\" length=\"height_armpit\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"B2\" type=\"endLine\" typeLine=\"hair\"/>",
          "armpit-90": "<point angle=\"90\" basePoint=\"6\" id=\"7\" length=\"height_armpit\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"B3\" type=\"endLine\" typeLine=\"hair\"/>",
          "armpit-270": "<point angle=\"270\" basePoint=\"4\" id=\"5\" length=\"height_armpit\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"B1\" type=\"endLine\" typeLine=\"hair\"/>",
          "waist-side-0": "<point angle=\"360\" basePoint=\"5\" id=\"6\" length=\"height_waist_side\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"B2\" type=\"endLine\" typeLine=\"hair\"/>",
          "waist-side-90":  "<point angle=\"90\" basePoint=\"6\" id=\"7\" length=\"height_waist_side\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"B3\" type=\"endLine\" typeLine=\"hair\"/>",
          "waist-side-270": "<point angle=\"270\" basePoint=\"4\" id=\"5\" length=\"height_waist_side\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"B1\" type=\"endLine\" typeLine=\"hair\"/>",
          "hip-0": "<point angle=\"360\" basePoint=\"5\" id=\"6\" length=\"height_hip\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"B2\" type=\"endLine\" typeLine=\"hair\"/>",
          "hip-90": "<point angle=\"90\" basePoint=\"6\" id=\"7\" length=\"height_hip\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"B3\" type=\"endLine\" typeLine=\"hair\"/>",
          "hip-270": "<point angle=\"270\" basePoint=\"4\" id=\"5\" length=\"height_hip\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"B1\" type=\"endLine\" typeLine=\"hair\"/>",
          "gluteal-fold-0": "<point angle=\"360\" basePoint=\"5\" id=\"6\" length=\"height_gluteal_fold\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"B2\" type=\"endLine\" typeLine=\"hair\"/>",
          "gluteal-fold-90": "<point angle=\"90\" basePoint=\"6\" id=\"7\" length=\"height_gluteal_fold\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"B3\" type=\"endLine\" typeLine=\"hair\"/>",
          "gluteal-fold-270": "<point angle=\"270\" basePoint=\"4\" id=\"5\" length=\"height_gluteal_fold\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"B1\" type=\"endLine\" typeLine=\"hair\"/>",
          "knee-0": "<point angle=\"360\" basePoint=\"5\" id=\"6\" length=\"height_knee\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"B2\" type=\"endLine\" typeLine=\"hair\"/>",
          "knee-90": "<point angle=\"90\" basePoint=\"6\" id=\"7\" length=\"height_knee\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"B3\" type=\"endLine\" typeLine=\"hair\"/>",
          "knee-270": "<point angle=\"270\" basePoint=\"4\" id=\"5\" length=\"height_knee\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"B1\" type=\"endLine\" typeLine=\"hair\"/>",
          "calf-0": "<point angle=\"360\" basePoint=\"5\" id=\"6\" length=\"height_calf\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"B2\" type=\"endLine\" typeLine=\"hair\"/>",
          "calf-90": "<point angle=\"90\" basePoint=\"6\" id=\"7\" length=\"height_calf\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"B3\" type=\"endLine\" typeLine=\"hair\"/>",
          "calf-270": "<point angle=\"270\" basePoint=\"4\" id=\"5\" length=\"height_calf\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"B1\" type=\"endLine\" typeLine=\"hair\"/>",
          "ankle-high-0": "<point angle=\"360\" basePoint=\"5\" id=\"6\" length=\"height_ankle_high\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"B2\" type=\"endLine\" typeLine=\"hair\"/>",
          "ankle-high-90": "<point angle=\"90\" basePoint=\"6\" id=\"7\" length=\"height_ankle_high\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"B3\" type=\"endLine\" typeLine=\"hair\"/>",
          "ankle-high-270": "<point angle=\"270\" basePoint=\"4\" id=\"5\" length=\"height_ankle_high\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"B1\" type=\"endLine\" typeLine=\"hair\"/>",
          "ankle-0": "<point angle=\"360\" basePoint=\"5\" id=\"6\" length=\"height_ankle\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"B2\" type=\"endLine\" typeLine=\"hair\"/>",
          "ankle-90": "<point angle=\"90\" basePoint=\"6\" id=\"7\" length=\"height_ankle\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"B3\" type=\"endLine\" typeLine=\"hair\"/>",
          "ankle-270": "<point angle=\"270\" basePoint=\"4\" id=\"5\" length=\"height_ankle\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"B1\" type=\"endLine\" typeLine=\"hair\"/>",
          "end-line": "<line firstPoint=\"7\" id=\"8\" lineColor=\"black\" secondPoint=\"4\" typeLine=\"hair\"/>"
    }[x]

def replaceFolder():
    dirItems = os.listdir(InputDirectory)
    if '.DS_Store' in dirItems: dirItems.remove('.DS_Store')
    files = [item for item in dirItems if os.path.isfile(os.path.join(InputDirectory, item))]
    files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    for count, file in enumerate(files):
        output_file = file[:-3] + ".val"
        compile(file, output_file)

def compile(input_file, output_file):
    editableOutput = open(OutputDirectory+output_file, "w")
    readableInput= open(InputDirectory+input_file, "r")

    for line in repeatLines:
        editableOutput.write(line + "\n")

    for counter, line in enumerate(readableInput):
            words = line.split()
            for word in words:
                print(RSLSub(word))
                editableOutput.write(RSLSub(word) + "\n")

    for line in repeatLines2:
        editableOutput.write(line + "\n")

if __name__ == "__main__":

    #argv = sys.argv[1:]
    #length = len(argv)
    #if length != 0:
    #    input_folder = argv[0]
    #else:
    #    print("Error: not enough argument supplied:")
    #    print("compiler.py <path> <file name>")
    #    exit(0)

    replaceFolder()
