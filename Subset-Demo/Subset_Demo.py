import sys
import os
from os import listdir
from os.path import isfile, join
import json
from collections import defaultdict
import cv2
import numpy as np
import math
import re
import random
from PIL import Image

degree_round = 10
distance_round = 25
xml_shrink = 50

master_Directory = '/Users/theodoreseem/res.Network/Subset-Demo/'
lineImage_mapping = '/Users/theodoreseem/res.Network/Subset-Demo/dictionaries/Line-Image.json'
lineExt_mapping = '/Users/theodoreseem/res.Network/Subset-Demo/dictionaries/Line-Extraction.json'

repeatLines = ["<?xml version=\"1.0\" encoding=\"UTF-8\"?>", "<pattern>", "<!--Pattern created with Seamly2D v0.6.0.0a (https://fashionfreedom.eu/).-->", "<version>0.6.0</version>",
                "<unit>inch</unit>", "<description/>","<notes/>","<measurements>/Users/theodoreseem/ResonanceHub/image2XML/image2XML/Data/Vector_Measurements/Tucker-Brianna.vit</measurements>",
                "<increments/>", "<draw name=\"Test1\">", "<calculation>", "<point id=\"1\" mx=\".1\" my=\".1\" name=\"A\" type=\"single\" x=\"90\" y=\"90\"/>",
                "<point angle=\"90\" basePoint=\"1\" id=\"2\" length=\"height*2\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"A1\" type=\"endLine\" typeLine=\"hair\"/>",
                "<point angle=\"180\" basePoint=\"1\" id=\"3\" length=\"height*2\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"A2\" type=\"endLine\" typeLine=\"hair\"/>",
                "<point id=\"4\" mx=\".1\" my=\".1\" name=\"B\" type=\"single\" x=\"20\" y=\"20\"/>"]
repeatLines2 = ["</calculation>","<modeling/>","<details/>","</draw>","</pattern>"]

class Feature:

    orderNum = ...;

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1; self.y1 = y1; self.x2 = x2; self.y2 = y2
        self.endpoints = (x1,y1),(x2,y2)

class Line(Feature):

    orderNum = ...;

    def __init__(self, x1, y1, x2, y2, order):

        def my_round(x, roundNum):
            return round(x/roundNum)*roundNum

        def calc_degrees(angle):
            if angle > 0:
                return(int(angle*57.2958))
            degree = int(360 - abs(angle*57.2958))
            if degree == 360: degree = 0
            return degree

        Feature.__init__(self,x1,y1,x2,y2)
        self.angle = round(math.atan2(y2-y1, x2-x1),2)*-1
        self.degrees = calc_degrees(self.angle)
        self.distance = my_round(int(math.sqrt((x2-x1)**2 + (y2-y1)**2)),5)
        self.orderNum = order
        self.endpoints = (x1,y1),(x2,y2)

def floor_round(x, roundNum):
    return round(math.floor(x/roundNum))*roundNum

def is_close(x1, y1, x2, y2, d):
    distance = int(math.sqrt((x2-x1)**2 + (y2-y1)**2))
    return(distance<d)

def loadDictionaries():
    with open(lineImage_mapping) as data_file:
        lineImg_mapping = json.load(data_file); lineImg_mapping = defaultdict(lambda: "NO IMG", lineImg_mapping)
    with open(lineExt_mapping) as ext_file:
        lExt_mapping = json.load(ext_file); lExt_mapping = defaultdict(lambda: "NO MEASURE", lExt_mapping)
    return lineImg_mapping, lExt_mapping

def clean_Image(image):

    img = cv2.imread(image)
    img_border =  cv2.copyMakeBorder(img,25,25,25,25,cv2.BORDER_CONSTANT,value=255)
    height, width, channels = img_border.shape
    new_image = np.ones((height,width,3), np.uint8)*255
    for i in range(0,height):
        for j in range(0,width):
            if np.any(img_border[i,j] > 4):
                img_border[i,j] = [255,255,255]
    gray = cv2.cvtColor(img_border,cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5),np.uint8)
    close = cv2.erode(gray, kernel, iterations = 2)
    ret, thresh = cv2.threshold(close, 100, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    contours_by_size = sorted(contour_sizes, key=lambda x: x[0])
    pattern_contour = contours_by_size[-2][1]
    cv2.drawContours(new_image, pattern_contour, -1, (0,0,0), 1)
    new_image = cv2.erode(new_image, kernel, iterations = 2)
    return new_image

def retrieveVertices(img):

    height, width, channels = img.shape
    erosion = cv2.dilate(img,np.ones((2,2),np.uint8),iterations = 1)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray,(5,5))

    # find Harris corners
    dst = cv2.cornerHarris(np.float32(gray),21,15,.1)  #3,11,.1
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,127,255,cv2.THRESH_BINARY)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst, connectivity=4)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, .01)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(10,10),(-1,-1),criteria)
    vertices = list(np.int0(corners))
    centerVertex = [v for v in vertices if is_close(v[0], v[1], (width/2), (height/2), 10)]
    if centerVertex:
        vertices.remove(centerVertex[0])
    for v in vertices:
        cv2.circle(img, (v[0],v[1]), 5, (0,0,255), -1)

    int = random.randint(0,100000)
    f = "/Users/theodoreseem/res.Network/Subset-Demo/test/%s.png" %(int)
    print(f)
    cv2.imwrite(f,img)

    return vertices

def createDAG(vertices):

    def get_dist(x1,y1,x2,y2):
        return int(math.sqrt((x2-x1)**2 + (y2-y1)**2))
    def clockwiseangle_and_distance(point):
        origin = [700, 700]
        refvec = [0, 1]
        vector = [point[0]-origin[0], point[1]-origin[1]]
        lenvector = math.hypot(vector[0], vector[1])
        if lenvector == 0:
            return -math.pi, 0
        normalized = [vector[0]/lenvector, vector[1]/lenvector]
        dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]
        diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]
        angle = math.atan2(diffprod, dotprod)
        if angle < 0:
            return 2*math.pi+angle, lenvector
        return angle, lenvector

    ccWise = sorted(vertices, key=clockwiseangle_and_distance)
    featureList = []
    for pos, v in enumerate(ccWise[:-1]):
        feat = Feature(ccWise[pos][0], ccWise[pos][1],ccWise[pos+1][0], ccWise[pos+1][1])
        featureList.append(feat)
    feat = Feature(ccWise[-1][0], ccWise[-1][1],ccWise[0][0], ccWise[0][1])
    featureList.append(feat)
    for count, f in enumerate(featureList):
        f.orderNum = count
    return featureList

def createXML(lines, output_file, height): #SUBSET FOR ONLY LINES

    editableOutput = open(output_file, "w")

    #PROBABLY NEED TO ALIGN THE STARTING POINT (4) TO THE INITIAL X Y VALUES FOR THE STARTING FEATURE
    startPoint = 4
    currentID = startPoint

    for line in lines:
        line.x1 = (line.x1/xml_shrink)+50; line.x2 = (line.x2/xml_shrink)+50; line.y1 = line.y1/xml_shrink; line.y2 = line.y2/xml_shrink; line.distance = line.distance/xml_shrink

    for line in repeatLines:
        editableOutput.write(line + "\n")

    for counter, line in enumerate(lines[:-1]):
        editableOutput.write("<point angle=\"%d\" basePoint=\"%d\" id=\"%d\" length=\"%.1f\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"B1\" type=\"endLine\" typeLine=\"hair\"/>\n" %(line.degrees, counter+startPoint, counter+startPoint+1, line.distance))
        currentID = currentID + 1
    editableOutput.write("<line firstPoint=\"%d\" id=\"%d\" lineColor=\"black\" secondPoint=\"%d\" typeLine=\"hair\"/>\n" %(currentID, currentID+1, startPoint))


    for line in repeatLines2:
        editableOutput.write(line + "\n")

    editableOutput.close()

def buildImageRepresentation(imageList, output_RIL):

    files = ['/Users/theodoreseem/res.Network/Hasy-images/' + image for image in imageList]

    result = Image.new("RGB", (500, 100))

    for index, file in enumerate(files):
        path = os.path.expanduser(file)
        img = Image.open(path)
        img.thumbnail((32, 32), Image.ANTIALIAS)
        x = 50 + (index * 35)
        y = 33#index % 2 * 32
        w, h = img.size
        #print('pos {0},{1} size {2},{3}'.format(x, y, w, h))
        result.paste(img, (x, y, x + w, y + h))

    result.save(os.path.expanduser(output_RIL))

def run_single(input_Image, output_XML, output_RIL, output_cleaned):
    imageCleaned = clean_Image(input_Image)

    int = random.randint(0,100000)
    f = "/Users/theodoreseem/res.Network/Subset-Demo/cleaned_images/%s.png" %(int)
    cv2.imwrite(f,imageCleaned)




    image_vertices = retrieveVertices(imageCleaned)
    DAG = createDAG(image_vertices)

#    lineList = []; imageList = []
#    for feature in DAG:
#        line = Line(feature.x1, feature.y1, feature.x2, feature.y2, feature.orderNum)
#        lineList.append(line)
#        lineMeasure = line_ext_map[str(floor_round(line.distance,distance_round))],int(abs(floor_round(line.degrees,degree_round)))
#        lineImage = line_Image_map[str(lineMeasure)]
#        print(lineMeasure)
#        imageList.append(lineImage)

    #SHORTCUT
#    if 'NO IMG' not in imageList:
#        cv2.imwrite(output_cleaned, imageCleaned)
#        createXML(lineList, output_XML, 700)
#        buildImageRepresentation(imageList, output_RIL)

def create_RSL_files(xml_directory, rsl_directory):
    files = [f[:-4] for f in listdir(xml_directory) if f != '.DS_Store']
    for file in files:
        xml_file = xml_directory + file + ".val"
        rsl_file_name = rsl_directory + file + ".gui"
        readableInput = open(xml_file, "r")
        editableOutput = open(rsl_file_name, "w")

        editableOutput.write("first-point\n")
        editableOutput.write("frame-one\n")
        editableOutput.write("frame-two\n")
        editableOutput.write("second-point\n")

        for counter, line in enumerate(readableInput):
            if counter > 14 and "<point" in line:
                angle = 0; length = 0
                words = line.split()
                for word in words:
                    if "angle" in word:
                        angle = re.findall('"([^"]*)"', word)[0]
                    if "length" in word:
                        length= re.findall('"([^"]*)"', word)[0]
                wordOutput = "RSL-%s-%s" %(str(floor_round(float(angle),degree_round)),str(floor_round(float(length)*xml_shrink, distance_round)))
                editableOutput.write(wordOutput)
                editableOutput.write('\n')

        editableOutput.write("end-line")
        editableOutput.close()
        readableInput.close()

def RSL_to_XML(rsl):
    xml = rsl[:-3] + "val"
    readableInput = open(rsl, "r")
    editableOutput = open(xml, "w")

    startPoint = 4
    currentID = startPoint

    for line in repeatLines:
        editableOutput.write(line + "\n")

    for counter, line in enumerate(readableInput, start=-4):
        if "RSL" in line:
            if line[5] == "-":
                angle = int(line[4:5])
            elif line[6] == "-":
                angle = int(line[4:6])
            else:
                angle = int(line[4:7])
            if line[-4] == "-":
                length = int(line[-3:])
            elif line[-6] == "-":
                length = int(line[-5:])
            else:
                length = int(line[-4:])
            editableOutput.write("<point angle=\"%d\" basePoint=\"%d\" id=\"%d\" length=\"%.1f\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"B1\" type=\"endLine\" typeLine=\"hair\"/>\n" %(angle, counter+startPoint, counter+startPoint+1, length/50))
            currentID = currentID + 1
    editableOutput.write("<line firstPoint=\"%d\" id=\"%d\" lineColor=\"black\" secondPoint=\"%d\" typeLine=\"hair\"/>\n" %(currentID, currentID+1, startPoint))


    for line in repeatLines2:
        editableOutput.write(line + "\n")

    editableOutput.close()



def createFiles(file, directory):
    print("Processing: ", file)
    input_Image = master_Directory + directory + "/" + file + ".JPEG"
    output_XML =  master_Directory + "xml_outputs/" + file + ".val"#argv[1]
    output_RIL =  master_Directory + "ril_outputs/" + file + ".png"#argv[2]
    output_cleaned =  master_Directory + "cleaned_images/" + file + ".png"#argv[2]
    run_single(input_Image, output_XML, output_RIL, output_cleaned)
    print("\n")



if __name__ == "__main__":

    argv = sys.argv[1:]
    arg_length = len(argv)
    if arg_length != 0:
        if argv[0] == "single":
            input = argv[1]
            files = [input]
        else:
            directory = argv[1]
            files = [f[:-5] for f in listdir(master_Directory + directory) if f != '.DS_Store']# if isfile(join(master_Directory+directory, f))]
    else:
        print("Error: not enough argument supplied: \n compiler.py <path> <file name>")
        exit(0)

    line_Image_map, line_ext_map = loadDictionaries()

    for count, file in enumerate(files):
        print(count)
        createFiles(file, directory)

#    create_RSL_files(master_Directory + "xml_outputs/", master_Directory + "rsl_outputs/")

#    RSL_to_XML("rsl_outputs/KT-6017-V-BODFT-2.gui")
