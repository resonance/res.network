import sys
import re
import os
import argparse
import random


#MAP_FILE = "/Users/theodoreseem/ResonanceHub/image2XML/image2XML/Hough-RNN/Line-RSL-map.json"

InputDirectory = "/Users/theodoreseem/Desktop/Emoji_Experiment/10-fold_rectangles_xml/"
OutputDirectory = "/Users/theodoreseem/Desktop/Emoji_Experiment/10-fold_RSL2/"

def replaceFolder(FileNum):
    dirItems = os.listdir(InputDirectory)
    if '.DS_Store' in dirItems: dirItems.remove('.DS_Store')
    files = [item for item in dirItems if os.path.isfile(os.path.join(InputDirectory, item))]
    files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    for count, file in enumerate(files):
        OutputFile = "test"+str(count+3)+".gui"
        replaceFile(file, OutputFile)

def replaceFile(inputFile, outputFile):
    print(inputFile)
    editableOutput = open(OutputDirectory+outputFile, "w")
    readableInput= open(InputDirectory+inputFile, "r")
    repeatLines = ["first-point\n","frame-one\n", "frame-two\n", "second-point\n"]
    #for line in repeatLines:
        #editableOutput.write(line)

    for counter, line in enumerate(readableInput):
        len = ""; angle = ""
        if counter > 14 and counter < 18:
            words = line.split()
            for word in words:
                if "length" in word:
                    len = re.findall('"([^"]*)"', word)[0]
                if "angle" in word:
                    ang = re.findall('"([^"]*)"', word)[0]
                    print(ang)
                    if ang == "360": angle = 360
                    elif ang == "270": angle = 270
                    elif ang == "180": angle = 180
                    else: angle = 90
            key = str((len,angle))
            print(key)
            wordOutput = lengthAngleSub(key)
            print(wordOutput)
            editableOutput.write(wordOutput+"\n")
    editableOutput.write("end-line")

def lengthAngleSub(x):
    return {
          "('height_neck_back', 360)":     "neck-back-0",
          "('height_neck_back', 90)":    "neck-back-90",
          "('height_neck_back', 270)":    "neck-back-270",
          "('height_neck_back', 180)":    "neck-back-180",
          "('height_scapula', 360)":       "scapula-0",
          "('height_scapula', 90)":      "scapula-90",
          "('height_scapula', 270)":      "scapula-270",
          "('height_scapula', 180)":      "scapula-180",
          "('height_armpit', 360)":        "armpit-0",
          "('height_armpit', 90)":       "armpit-90",
          "('height_armpit', 270)":       "armpit-270",
          "('height_armpit', 180)":       "armpit-180",
          "('height_waist_side', 360)":    "waist-side-0",
          "('height_waist_side', 90)":   "waist-side-90",
          "('height_waist_side', 270)":   "waist-side-270",
          "('height_waist_side', 180)":   "waist-side-180",
          "('height_hip', 360)":           "hip-0",
          "('height_hip', 90)":          "hip-90",
          "('height_hip', 270)":          "hip-270",
          "('height_hip', 180)":          "hip-180",
          "('height_gluteal_fold', 360)":  "gluteal-fold-0",
          "('height_gluteal_fold', 90)": "gluteal-fold-90",
          "('height_gluteal_fold', 270)": "gluteal-fold-270",
          "('height_gluteal_fold', 180)": "gluteal-fold-180",
          "('height_knee', 360)":          "knee-0",
          "('height_knee', 90)":         "knee-90",
          "('height_knee', 270)":         "knee-270",
          "('height_knee', 180)":         "knee-180",
          "('height_calf', 360)":          "calf-0",
          "('height_calf', 90)":         "calf-90",
          "('height_calf', 270)":         "calf-270",
          "('height_calf', 180)":         "calf-180",
          "('height_ankle_high', 360)":    "ankle-high-0",
          "('height_ankle_high', 90)":   "ankle-high-90",
          "('height_ankle_high', 270)":   "ankle-high-270",
          "('height_ankle_high', 180)":   "ankle-high-180",
          "('height_ankle', 360)":         "ankle-0",
          "('height_ankle', 90)":        "ankle-90",
          "('height_ankle', 270)":        "ankle-270",
          "('height_ankle', 180)":        "ankle-180"
    }[x]


if __name__ == "__main__":

        replaceFolder(1)
