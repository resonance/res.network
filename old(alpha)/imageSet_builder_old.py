import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from numpy import *
from scipy import stats
import scipy.misc
import copy
from sklearn.cluster import KMeans
from PIL import Image, ImageFont, ImageDraw, TiffImagePlugin
import io
from pdf2image import convert_from_path
import os
import time
import json
import re
from emojipy import Emoji

Emoji.unicode_alt = False

#Need to output number of lines in file (get number of features from image name)
#Need better feature extraction so we don't have duplicate lines

##############External Images####################

patterns_folder = '/Users/theodoreseem/Desktop/Emoji2Code/10-fold_images_cropped/'
featureImage_mapping = '/Users/theodoreseem/Desktop/Emoji2Code/dictionaries/Feature-Image_mapping.json'
extraction_mapping = '/Users/theodoreseem/Desktop/Emoji2Code/dictionaries/extraction_mapping.json'
PNG_path = '/Users/theodoreseem/Desktop/Emoji2Code/ResImages/'
#-----
DEBUG = False
debug_output_path = '/Users/theodoreseem/Desktop/Emoji2Code/ExtractedPatterns/'
#-----
font_file = '/Users/theodoreseem/Desktop/Emoji_Experiment/Symbola/Symbola_hint.ttf'

#################################################

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

class Line(object):

    slope = ...; intercept = ...; xcept = ...; x1 = ...; y1 = ...
    angle = ...; degrees = ...;  distance = ...; x2 = ...; y2 = ...

    def __init__(self, slope, intercept, xcept, angle, degrees, distance, x1, y1, x2, y2):
        self.slope = slope; self.intercept = intercept; self.xcept = xcept; self.x1 = x1; self.y1 = y1
        self.angle = angle; self.degrees = degrees; self.distance = distance; self.x2 = x2; self.y2 = y2

def createLines(lines):
    lineObjects = []
    for line in lines:
        s,i,x,a,d,dis = get_stats(line)
        for x1,y1,x2,y2 in line:
            lineObject = Line(s,i,x,a,d,dis,x1,y1,x2,y2)
        lineObjects.append(lineObject)
    return lineObjects

def __sim__(self, other):
    return withinLimit(self.angle,other.angle,.1) and withinLimit(self.xcept,other.xcept,50)\
        and withinLimit(self.intercept,other.intercept,50) and withinLimit(self.distance,other.distance,50)\
        and self != other

def simCompare(x,list):
    for i in list:
        if x.__eq__(i):
            return True
    return False

def Remove(full_list):
    final_list = []
    for num in full_list:
        if num not in final_list:
            final_list.append(num)
    return final_list

def get_stats(line):
    for x1,y1,x2,y2 in line:
        x_x = [x1,x2]; y_y = [y1, y2]
        y = y2-y1; x = x2-x1
        angle = math.atan2(y, x)
        distance = my_round(int(math.sqrt((x2-x1)**2 + (y2-y1)**2)),5)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_x,y_y)
        degrees = angle*57.2958
        if isnan(slope): slope = 999999
        else: slope = round(slope, 1)
        if slope == 0: xcept = 999999
        else: xcept = my_round((-y1/slope)+x1,10)
        if slope == 999999: intercept = 999999
        else: intercept = my_round(intercept,50)
        angle = round(angle,2); degrees = my_round(degrees,5)
    return slope, intercept, xcept, angle, degrees, distance

def withinLimit(num1,num2,lim):
    inLimit = False
    if num1<=num2+lim and num1>=num2-lim or num1 == num2:
        inLimit = True
    print("in limit", num1, num2, inLimit)
    return inLimit

def my_round(x, roundNum):
    return round(x/roundNum)*roundNum

def createResImage(height, width, count, lines):
    image = np.ones((height, width,3))*255
    PNG = PNG_path+'test'+str(count)+'.png'
    distMeasures = [(ext_mapping[str(line.distance)],int(abs(line.degrees))) for line in lines if str(line.distance) in ext_mapping]
    emojis = [featImg_mapping[str(distMeasure)]for distMeasure in distMeasures]
    print(distMeasures); print(emojis)
    text = ""
    for emoji in emojis[0:4]: #NEED BETTER FILTERING TO ONLY INCLUDE 4 WE WANT
        text = text + " " + emoji
    image = Image.new("RGBA", (1000,300), (255,255,255))
    font = ImageFont.truetype(font_file, 15, encoding='unic')
    draw = ImageDraw.Draw(image)
    draw.text((0,0), text, (0,0,0), font=font)
    image.save(PNG)

def processLines(lines):
    #printLines(lines, "normal Lines")
    orderedLines, firstLine = extractDAG(lines)
    #printLines(orderedLines, "ordered Lines")
    coordswappedLines = coordinateRecompute(orderedLines, firstLine)
    #printLines(coordswappedLines, "cord swapped Lines")
    angleComputeLines = angleRecompute(orderedLines)
    #printLines(angleComputeLines, "angle fixed Lines")
    return angleComputeLines

#gets the directionality of lines and also re-assignes coordinates to the DAG direction (all lines flow same direction as DAG)
def extractDAG(lines):
    realLines = [line for line in lines if str(line.distance) in ext_mapping]
    currentLine = get_start_coord(0,0,realLines)
    firstLine = currentLine
    linesOrdered = [currentLine[0]]
    for lineNum in range(len(realLines)-1):
        currentLine = get_next_line(currentLine[0], oppositeCord(currentLine[1]), realLines)
        linesOrdered.append(currentLine[0])
    return linesOrdered, firstLine

def oppositeCord(num):
    if num == 1: return 2
    else: return 1

def get_dist(x1,y1,x2,y2):
    return int(math.sqrt((x2-x1)**2 + (y2-y1)**2))

def get_start_coord(x, y, lines):
    closest = ...; shortDist = 10000
    for line in lines:
        if get_dist(line.x1,line.y1,x,y) < shortDist:
            shortDist = get_dist(line.x1,line.y1,x,y)
            closest = (line, 1)
        if get_dist(line.x2,line.y2,x,y) < shortDist:
            shortDist = get_dist(line.x2,line.y2,x,y)
            closest = (line, 2)
    return closest #returns line and which coordinate

#Finds the closest end-coordinate of all lines to the previous coordinate. Returns it's line presence.
def get_next_line(inputLine, coord, lines):
    if coord == 1: x, y = inputLine.x1, inputLine.y1
    else: x, y = inputLine.x2, inputLine.y2
    closest = ...; shortDist = 10000
    for line in lines:
        if line != inputLine:
            if get_dist(line.x1,line.y1,x,y) <= shortDist:
                shortDist = get_dist(line.x1,line.y1,x,y)
                closest = (line, 1) #return line and which coordinate was closest
            if get_dist(line.x2,line.y2,x,y) <= shortDist:
                shortDist = get_dist(line.x2,line.y2,x,y)
                closest = (line, 2) #return line and which coordinate closest
    return closest #returns line and which coordinate

def coordinateRecompute(orderedLines, firstLine): #first line is line and which coordinate x1,y1, or x2,y2 is closer
    if firstLine[1] == 2: orderedLines[0] = swapCoordinates(orderedLines[0])
    for pos, line in enumerate(orderedLines, 1):
        if pos < len(orderedLines):
            if get_dist(orderedLines[pos-1].x2, orderedLines[pos-1].y2, orderedLines[pos].x1, orderedLines[pos].y1) > 10:
                orderedLines[pos] = swapCoordinates(orderedLines[pos])
    return orderedLines

def swapCoordinates(line):
    hold_x = line.x1; hold_y = line.y1
    line.x1 = line.x2; line.y1 = line.y2
    line.x2 = hold_x; line.y2 = hold_y
    return line

#NEED TO COME BACK TO THIS, BECAUSE DIRECTIONALITY OF DAG IS arbitrary to first coordinate.
def angleRecompute(oLines):
    for line in oLines:
        x = line.x2-line.x1; y = line.y2-line.y1
        angle = math.atan2(y,x)
        line.angle = angle
        line.degrees = int(line.angle*57.2958)
        if line.degrees == 90: line.degrees = 270
        if line.degrees == -90: line.degrees = 90
    return oLines

def printLines(lines, name):
    print(name)
    for l in lines:
        print(l.distance, l.angle, l.degrees, l.x1, l.y1, l.x2, l.y2)

def remove_similar(lines):
    lineFeatures = []; fin_lines = []
    lineObjects = createLines(lines)
    lineObjects = [line for line in lineObjects if line.distance<1500]

    for i in lineObjects:
        lineFeatures.append([i.slope,i.xcept,i.intercept])
    kmeans = KMeans(n_clusters=len(lineObjects))
    kmeans.fit(lineFeatures)
    y_kmeans = kmeans.predict(lineFeatures)
    for cluster in range(0,len(lineObjects)):
        x1 = 0 ; y1 = 0; x2 = 0; y2 = 0
        contributors = 0
        for pos, line in enumerate(lineObjects):
            if y_kmeans[pos] == cluster:
                x1 = x1+line.x1; y1 = y1+line.y1; x2 = x2+line.x2; y2 = y2+line.y2
                contributors = contributors + 1
        x1 = int(x1/max(1,contributors)); y1 = int(y1/max(1,contributors))
        x2 = int(x2/max(1,contributors)); y2 = int(y2/max(1,contributors))
        s,i,x,a,d,dis = get_stats(np.asarray([np.array([x1, y1, x2, y2])]))
        fin_lines.append(Line(s,i,x,a,d,dis,x1,y1,x2,y2))
    return fin_lines

def outputImage(height,width,count,lineObjs):
    line_image = np.ones((height, width,3))*255
    for line in lineObjs:
        cv2.line(line_image,(line.x1,line.y1),(line.x2,line.y2),(255, 0, 0),1)
    print(writtenName); time.sleep(.5)
    cv2.imwrite(debug_output_path+'test'+str(count)+'.jpg',line_image)

def extract_features(image):
    img = cv2.imread(image)
    for x in range(1150,1275):
        for y in range(550,675):
            img[y,x] = [255,255,255]
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge_img = cv2.Canny(gray_img,230,250,apertureSize = 5)
    blur_img = cv2.blur(edge_img,(2,2))
    height, width= edge_img.shape

    threshold = 30
    rho = 2; theta = np.pi/18; min_line_length = 30; max_line_gap = 30

    lines = cv2.HoughLinesP(edge_img, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    lineObjs = remove_similar(lines)
    return height, width, lineObjs


def convertPatterns(folder):
    items = os.listdir(folder)
    if '.DS_Store' in items: items.remove('.DS_Store')
    files = [os.path.join(folder, item) for item in items]
    files = natural_sort(files)
    for count, file in enumerate(files):
        if count < len(files):
            print(file[-10:])
            height,width,lines = extract_features(file)
            processed_lines = processLines(lines)
            if DEBUG: outputImage(height,width,count+3,processed_lines)
            createResImage(height, width, count+3, processed_lines)


######### MAIN FUNCTION  ################

if __name__ == "__main__":

    with open(featureImage_mapping) as data_file:
        featImg_mapping = json.load(data_file)
    with open(extraction_mapping) as ext_file:
        ext_mapping = json.load(ext_file)

    convertPatterns(patterns_folder)

    #testImage_name = '/Users/theodoreseem/Desktop/Emoji_Experiment/InputPattern.jpg'
    #lines = extract_lines(testImage_name)
