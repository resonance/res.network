import sys
import os
from os import listdir
from os.path import isfile, join
import json
from collections import defaultdict
import cv2
import numpy as np
import math
import numpy.polynomial.polynomial as poly
from skimage import util
import re
from PIL import Image
import statistics
import random
np.set_printoptions(threshold=np.nan)

## TODO: Fix the placement of Corners (use overlap of dst and curve and then find centroid), need notches
## TODO: Add basic curve Curve integration

degree_round = 10
distance_round = 25
xml_shrink = 50

master_Directory = '/Users/theodoreseem/res.Network/Curve_Integration/'

lineImage_mapping = '/Users/theodoreseem/res.Network/Curve_Integration/dictionaries/Line-Image.json'
featureExt_mapping = '/Users/theodoreseem/res.Network/Curve_Integration/dictionaries/Feature-Extraction.json'
curveImage_mapping = '/Users/theodoreseem/res.Network/Curve_Integration/dictionaries/Curve-Image.json'

repeatLines = ["<?xml version=\"1.0\" encoding=\"UTF-8\"?>", "<pattern>", "<!--Pattern created with Seamly2D v0.6.0.0a (https://fashionfreedom.eu/).-->", "<version>0.6.0</version>",
                "<unit>inch</unit>", "<description/>","<notes/>","<measurements>/Users/theodoreseem/ResonanceHub/image2XML/image2XML/Data/Vector_Measurements/Tucker-Brianna.vit</measurements>",
                "<increments/>", "<draw name=\"Test1\">", "<calculation>", "<point id=\"1\" mx=\".1\" my=\".1\" name=\"A\" type=\"single\" x=\"90\" y=\"90\"/>",
                "<point angle=\"90\" basePoint=\"1\" id=\"2\" length=\"height*2\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"A1\" type=\"endLine\" typeLine=\"hair\"/>",
                "<point angle=\"180\" basePoint=\"1\" id=\"3\" length=\"height*2\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"A2\" type=\"endLine\" typeLine=\"hair\"/>",
                "<point id=\"4\" mx=\".1\" my=\".1\" name=\"B\" type=\"single\" x=\"20\" y=\"20\"/>"]
repeatLines2 = ["</calculation>","<modeling/>","<details/>","</draw>","</pattern>"]

class Feature:

    orderNum = ...;
    points = []
    filteredPoints = []

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1; self.y1 = y1; self.x2 = x2; self.y2 = y2
        self.endpoints = (x1,y1),(x2,y2)
        self.distance = my_round(int(math.sqrt((x2-x1)**2 + (y2-y1)**2)),5)

class Line(Feature):

    orderNum = ...;

    def __init__(self, x1, y1, x2, y2, order):

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

class Curve(Feature):

    orderNum = ...;

    def __init__(self, x1, y1, x2, y2, x3, y3, order):

        def my_round(x, roundNum):
            return round(x/roundNum)*roundNum

        def calc_degrees(angle):
            if angle > 0:
                return(int(angle*57.2958))
            return(int(360 - abs(angle*57.2958)))

        Feature.__init__(self,x1,y1,x3,y3)

        #second angle calc due to control points order
        self.angle1 = round(math.atan2(y2-y1, x2-x1),2)*-1; self.distance1 = my_round(int(math.sqrt((x2-x1)**2 + (y2-y1)**2)),5);
        self.angle2 = round(math.atan2(y2-y3, x2-x3),2)*-1; self.distance2 = my_round(int(math.sqrt((x2-x3)**2 + (y2-y3)**2)),5);
        self.degrees1 = calc_degrees(self.angle1);
        self.degrees2 = calc_degrees(self.angle2)
        self.orderNum = order;
        self.endpoints = (x1,y1),(x3,y3)

class Point:

    def __init__(self, x, y, cluster):
        self.x = x; self.y = y; self.cluster = cluster

def my_round(x, roundNum):
    return round(x/roundNum)*roundNum

def collinear_exact(Points):
    Points.sort(key=lambda x: x[0])
    x1, y1 = Points[0]
    x2, y2 = Points[len(Points)-1]
    collinearity = []
    for point in Points[1:-1]:
        x3, y3 = point
        area = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
        if area!=0:
            return False
    return True
def collinear_approx(Points):
    Points.sort(key=lambda x: x[0])
    x1, y1 = Points[0]
    x2, y2 = Points[len(Points)-1]
    x_diff = x2 - x1
    y_diff = y2 - y1
    for p in Points[1:-1]:
        x3, y3 = p
        num = abs(y_diff*x3 - x_diff*y3 + x2*y1 - y2*x1)
        den = math.sqrt(y_diff**2 + x_diff**2)
        dist = num/den
        if dist>8: return False
    return True

def findAssociatedFeature(contour, featureList):
    for f in featureList:
        coord_set = [(f.x1, f.y1),(f.x2, f.y2)]
        matches = []
        for coord in coord_set:
            for cont in contour:
                if is_close(coord[0], coord[1], cont[0], cont[1], 20):
                    matches.append(1)
                    break; # Only possible to find a single match per vertex (so needs 2 matches)
        if len(matches)==2:
            return f
    print("Internal Error: associated feature not found")
    exit(0)

def floor_round(x, roundNum):
    return round(math.floor(x/roundNum))*roundNum

def is_close(x1, y1, x2, y2, d):
    distance = int(math.sqrt((x2-x1)**2 + (y2-y1)**2))
    return(distance<d)

def loadDictionaries():
    with open(lineImage_mapping) as data_file:
        lineImg_mapping = json.load(data_file); lineImg_mapping = defaultdict(lambda: "NO IMG", lineImg_mapping)
    with open(featureExt_mapping) as ext_file:
        fExt_mapping = json.load(ext_file); fExt_mapping = defaultdict(lambda: "NO MEASURE", fExt_mapping)

    with open(curveImage_mapping) as data_file:
        curveImg_mapping = json.load(data_file); curveImg_mapping = defaultdict(lambda: "NO IMG", curveImg_mapping)
    return lineImg_mapping, fExt_mapping, curveImg_mapping

def clean_Image(image):

    '''Reads image, creates a blank copy for later, adds a white border to reduce image up against edges and copies every non-fully black pixel to white pixels to reduce noise'''
    img = cv2.imread(image)
    img_border =  cv2.copyMakeBorder(img,25,25,25,25,cv2.BORDER_CONSTANT,value=[255,255,255])
    height, width, channels = img_border.shape
    new_image = np.ones((height,width,3), np.uint8)*255

    mask = (img_border[:,:,0] > 6 ) & (img_border[:,:,1] > 6) & (img_border[:,:,2] > 6)
    img_border[:,:,:3][mask] = [255, 255, 255]

    ''' Converts image to gray scale (eliminates images third channel dimension), thickens the black lines, finds all contours in the image'''
    gray = cv2.cvtColor(img_border,cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5),np.uint8)
    close = cv2.erode(gray, kernel, iterations = 2)
    ret, thresh = cv2.threshold(close, 100, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    '''Sorts the contours by size, finds the biggest and redraws it  on the blank image created earlier, returns image'''
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    contours_by_size = sorted(contour_sizes, key=lambda x: x[0])
    pattern_contour = contours_by_size[-2][1]
    cv2.drawContours(new_image, pattern_contour, -1, (0,0,0), 1)
    new_image = cv2.erode(new_image, kernel, iterations = 2)
    return new_image

def retrieveVertices(img):

    '''get image dimensions, thin the pattern lines out on the image, convert to grayscale and blur (all optimizes corner detection) '''

    height, width, channels = img.shape
    erosion = cv2.dilate(img,np.ones((8,8),np.uint8),iterations = 1)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray,(2,2))

    '''Find the corner orientation (generates blobs of pixels where the corner orients), thins out the blob area and thresholds to compact the area'''
    dst = cv2.cornerHarris(np.float32(gray),21,9,.1)  #3,11,.1
    dst = cv2.erode(dst,np.ones((3,3),np.uint8),None)
    ret, dst = cv2.threshold(dst,127,255,cv2.THRESH_BINARY)
    dst = np.uint8(dst)

    '''For each blob representing a corner area/alignment, we find the centroid of that blob to accuratly place the corner'''
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst, connectivity=4)

    '''define the criteria to stop and refine the corners, and fitler out the center vertex which seems to sometimes appear in the middle of the image'''
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, .01)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(10,10),(-1,-1),criteria)
    vertices = list(np.int0(corners))
    centerVertex = [v for v in vertices if is_close(v[0], v[1], (width/2), (height/2), 10)]
    if centerVertex:
        vertices.remove(centerVertex[0])

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

    '''Sortes the vertices based on their clockwise orientation to the center using above fucntion as key'''
    ccWise = sorted(vertices, key=clockwiseangle_and_distance)

    '''Create a feature list (a feature is an implicit line connecting two endpoints(vertices)) and creates ordered list based on the vertices ordering'''
    featureList = []
    for pos, v in enumerate(ccWise[:-1]):
        feat = Feature(ccWise[pos][0], ccWise[pos][1],ccWise[pos+1][0], ccWise[pos+1][1])
        featureList.append(feat)
    feat = Feature(ccWise[-1][0], ccWise[-1][1],ccWise[0][0], ccWise[0][1])
    featureList.append(feat)
    for count, f in enumerate(featureList):
        f.orderNum = count
    return featureList

def extract_initial_contours(DAG, img):

    '''Get the endpoints of the features, thicken image lines and convert to grayscale, then slightly thicken lines and threshold the image'''
    vertices = [v.endpoints[0] for v in DAG]
    height, width, channels = img.shape
    erosion = cv2.dilate(img,np.ones((8,8),np.uint8),iterations = 1)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_conv = util.invert(gray)
    img_conv = cv2.erode(img_conv, np.ones((3,3),np.uint8), iterations = 2)
    ret, thresh = cv2.threshold(img_conv, 100, 255, 0)

    '''find the contours of the image (aka all points along the pattern lines) and convert the numpy array to tuple coordinates'''
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contourArray = np.vstack(contours[1]).squeeze()
    contourPoints = list(map(tuple, contourArray))

    def isCloseToVert(point):
        for v in vertices:
            if is_close(point[0], point[1], v[0], v[1], 15):
                return True
        return False

    '''filter out all points within 15 (hardcoded in above function) pixels of a vertex'''
    points_without_vertices = [p for p in contourPoints if not isCloseToVert(p)]

    '''create a new blank image, and draw a circle on the image for every contour point'''
    nonVertex_image = np.ones([height, width, 3],np.uint8)*255
    for p in points_without_vertices:
        cv2.circle(nonVertex_image, (p[0], p[1]), 1, (0,0,0), -1)

    '''thin out contour dots creating the lines, convert to grayscale and threshold the image again, then invert the colors so lines are white'''
    erosion = cv2.erode(nonVertex_image,np.ones((2,2),np.uint8),iterations = 1)
    graynonVert = cv2.cvtColor(erosion,cv2.COLOR_BGR2GRAY)
    threshnonVert = cv2.threshold(graynonVert, 127, 255, cv2.THRESH_BINARY)[1]
    inverted = 255-threshnonVert
    img_binary = np.array(inverted)

    ''' At this point we have an inverted image with white lines and 'gaps' with a radius of of 5 around the vertices'''
    ''' We can now do point clustering where points within 4 pixels of another are considered the same cluster, this is how we divide up the actual contours which belong to each feature'''
    n_labels, img_labeled, lab_stats, _ = cv2.connectedComponentsWithStats(img_binary, connectivity=4, ltype=cv2.CV_32S)


    #TODO: Change this to appending points to the correct DAG feature instead of creating two dimensional array

    '''This is the fucntion to take the clustered output from previous line and organize into arrays of clustered points'''
    '''Storing in two dimensional array where the first position of each second dimension holds all the points coordinates'''
    '''Said differently componentArray[0][0] holds coordinates of first feature, componentArray[1][0] the second feature, componentArray[2][0] the third...'''
    filledFeatures = np.empty_like(DAG)

    for component in range(1,len(lab_stats)):
        indices = np.where(img_labeled == component)
        indices = tuple(zip(*indices))
        indices = [t[::-1] for t in indices]

        feature = findAssociatedFeature(indices, DAG)
        feature.points = indices
        filledFeatures.append(feature)
        #componentArray[component-1].append(indices)

    #return componentArray
    return filledFeatures

def subdivisionFeatures(features):

    def iswWithin(bx1, bx2, by1, by2, p):
        if bx1<p[0]<bx2 and by1<p[1]<by2:
            return True
        return False
    def filter_points(points, step_size = 50):

        filtered_points = []
        smallestX = min(points, key=lambda x:x[0])[0]; largestX = max(points, key=lambda x:x[0])[0]
        smallestY = min(points, key=lambda x:x[1])[1]; largestY = max(points, key=lambda x:x[1])[1]
        for x in range(smallestX,largestX,step_size):
            for y in range(smallestY,largestY,step_size):
                subPoints = []
                for p in points:
                    if iswWithin(x, x+step_size, y, y+step_size, p):
                        subPoints.append(p)
                xs = [point[0] for point in subPoints]
                ys = [point[1] for point in subPoints]
                if xs:
                    avg_x = statistics.mean(xs)
                if ys:
                    avg_y = statistics.mean(ys)
                else:
                    avg_x, avg_y = 0,0
                if avg_x != 0 and avg_y !=0: filtered_points.append((avg_x, avg_y))
        return filtered_points
    def closestTo(anchor, points):
        closestDist = int(math.sqrt((points[0][0]-anchor[0])**2 + (points[0][1]-anchor[1])**2))
        closestPoint = points[0]
        for p in points:
            distToAnchor = int(math.sqrt((p[0]-anchor[0])**2 + (p[1]-anchor[1])**2))
            if distToAnchor < closestDist:
                closestDist = distToAnchor
                closestPoint = p
        return closestPoint

    subArray = np.empty_like(features)

    ''' Iterate through each feature and find the filtered points of it's contour points. This is done by evaluating 50x50 area dimensions of that feature area
        and averaging all points which fall into that 50x50 grid location
        If a feature's length is less than some (200) pixels in length, the sub pixels are populated as the endpoints (because no need to divide feature further)
    '''

    for f in range(len(features)):
        if f.distance > 200:
            f.filteredPoints  = filter_points(f.points, 50)
            AveragePoint = filter_points(f.points, 100000)
            closest_toAvg = (AveragePoint, f.filteredPoints)
            #Either divide up the contour points or go through the same process and create vertex, remove points, find contours, and do componentclustering



        else:
            f.filteredPoints = f.endpoints



    #Take the closest_toAvg point and use it to divide the array in half creating two new features replacing the old one in the DAG

    #return filteredArray, subArray

def calc_quadratic_bezier(polynomial, x1, x2):

    derivative = np.polyder(polynomial, 1)

    y1 = polynomial(x1);
    y1_prime = derivative(x1)
    y2 = polynomial(x2);
    y2_prime = derivative(x2);

    # Find intersection of tangents
        # line0: y - y0 = y0p * (x - x0)
        # line1: y - y1 = y1p * (x - x1)
        # line0: y = y0p * x - y0p * x0 + y0
        # line1: y = y1p * x - y1p * x1 + y1
        # y0p * x - y0p * x0 + y0 = y1p * x - y1p * x1 + y1
        # y0p * x - y1p * x = y0p * x0 - y0 - y1p * x1 + y1
        # x = (y0p * x0 - y0 - y1p * x1 + y1) / (y0p - y1p)

    # Intersection point of tangents
    x_cp = (y2_prime * x2 - y2 - y1_prime * x1 + y1) / (y2_prime - y1_prime);
    y_cp = y2_prime * x_cp - y2_prime * x2 + y2;
    return int(x_cp), int(y_cp)

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

def createImageList(initialFeatures, DAG):

    final_featuresList = []

    for f in range(len(initialFeatures)):
        featurePoints = initialFeatures[f][0]
        feature = findAssociatedFeature(featurePoints, DAG)
        if collinear_approx(featurePoints):
            lineObj = Line(feature.x1, feature.y1, feature.x2, feature.y2, feature.orderNum)
            final_featuresList.append(lineObj)
        else:
            xs = [x for x,y in featurePoints]; ys = [y for x,y in featurePoints]
            if abs(featurePoints[0][0]-featurePoints[int(len(featurePoints)/2)][0])<50 and abs(featurePoints[0][0]-featurePoints[-1][0])<50: #Assumption
                tmp = xs, ys
                ys = tmp[0]; xs = tmp[1]
            polynomial = np.poly1d(np.polyfit(xs, ys, 3))
            ys = [polynomial(x) for x in xs]
            x_cp, y_cp = calc_quadratic_bezier(polynomial,feature.x1,feature.x2)
            curveObj = Curve(feature.x1, feature.y1, x_cp, y_cp, feature.x2, feature.y2, feature.orderNum)
            final_featuresList.append(curveObj)

    imageList = []
    for finFeature in final_featuresList:
        if type(finFeature) == lineObj:
            featureMeasure = feat_ext_map[str(floor_round(lineObj.distance,distance_round))],int(abs(floor_round(lineObj.degrees,degree_round)))
            lineImage = line_Image_map[str(featureMeasure)]
            print(featureMeasure)
            imageList.append(lineImage)
        else:
            featureMeasure = feat_ext_map[str(floor_round(curveObj.distance2,distance_round))],int(abs(floor_round(curveObj.degrees2,degree_round)))
            curveImage = curve_Image_map[str(featureMeasure)]
            print(featureMeasure)
            imageList.append(curveImage)

    return imageList

def cleanAndBuild(file, imageStore):


    input_Image = master_Directory + imageStore + "/" + file + ".JPEG"
    output_XML =  master_Directory + "xml_outputs/" + file + ".val"
    output_RIL =  master_Directory + "ril_outputs/" + file + ".png"
    output_cleaned =  master_Directory + "cleaned_images/" + file + ".png"

    imageCleaned = clean_Image(input_Image)
    image_vertices = retrieveVertices(imageCleaned)
    DAG = createDAG(image_vertices)
    initialFeatures = extract_initial_contours(DAG, imageCleaned)
    refinedFeatures, delineatedFeatures = subdivisionFeatures(initialFeatures)



    height, width, channels = imageCleaned.shape
    new_image = np.ones((height,width,3), np.uint8)*255
    for f in range(len(initialFeatures)):
        featurePoints = initialFeatures[f][0]
        refinedPoints = refinedFeatures[f][0]
        delPoints = delineatedFeatures[f][0]

        for p in featurePoints:
            cv2.circle(new_image, (p[0], p[1]), 1, (255,0,0), -1)
        for m in refinedPoints:
            cv2.circle(new_image, (m[0], m[1]), 1, (0,255,0), -1)
        for m in delPoints:
            cv2.circle(new_image, (m[0], m[1]), 10, (0,0,255), -1)
    assignment = random.randint(0, 1000)
    cv2.imwrite('/Users/theodoreseem/res.Network/Curve_Integration/test/test' + str(assignment) + '.png', new_image)



    #imageList = createImageList(initialFeatures, DAG)


    #SHORTCUT
    #if 'NO IMG' not in imageList:
    #cv2.imwrite(output_cleaned, imageCleaned)
        #createXML(lineList, output_XML, 700)
        #buildImageRepresentation(imageList, output_RIL)



if __name__ == "__main__":

    '''
        Dynamically take in two arguments 1) type of arguments and 2) locational arguments; the code
        allows a single file to be processed or an entire to be processed.

        Argument1: <single/multi> - Determines whether you process one file or many
        Argument2: <location> This is either the file's name or the folder location of multiple documents.
                              Single file must be stand alone within master directory ending in JPEG
    '''

    argv = sys.argv[1:]
    arg_length = len(argv)
    if arg_length != 0:
        if argv[0] == "single":
            imgStore = argv[1]
        else:
            imgStore = argv[1]
            files = [f[:-5] for f in listdir(master_Directory + imgStore) if f != '.DS_Store']
    else:
        print("Error: not enough argument supplied: \n Processing.py <single/multi> <Image/Image_Directory>")
        exit(0)

    line_Image_map, feat_ext_map, curve_Image_map = loadDictionaries()

    if argv[0] == "single":
        file = imgStore[:-5]
        print("Processing: ", file)
        cleanAndBuild(file, "")
    else:
        for count, file in enumerate(files[:20]):
            print("#", count, "- Processing: ", file)
            cleanAndBuild(file, imgStore)

#    create_RSL_files(master_Directory + "xml_outputs/", master_Directory + "rsl_outputs/")

#    RSL_to_XML("rsl_outputs/KT-6017-V-BODFT-2.gui")
