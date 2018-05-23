import sys
import cv2
import numpy as np
import math
import json
import os
from skimage import util
from collections import defaultdict
from matplotlib import *
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
from scipy.spatial import distance as dist
from sympy import Matrix
import scipy.ndimage, scipy.interpolate
from PIL import Image

from pylab import randn


lineImage_mapping = '/Users/theodoreseem/res.Network/dictionaries/Line-Image_mapping.json'
curveImage_mapping = '/Users/theodoreseem/res.Network/dictionaries/curveImage_mapping.json'
lineExt_mapping = '/Users/theodoreseem/res.Network/dictionaries/lineExtraction_mapping.json'
curveExt_mapping = '/Users/theodoreseem/res.Network/dictionaries/curveExtraction_mapping.json'

repeatLines = ["<?xml version=\"1.0\" encoding=\"UTF-8\"?>", "<pattern>", "<!--Pattern created with Seamly2D v0.6.0.0a (https://fashionfreedom.eu/).-->", "<version>0.6.0</version>",
                "<unit>inch</unit>", "<description/>","<notes/>","<measurements>/Users/theodoreseem/ResonanceHub/image2XML/image2XML/Data/Vector_Measurements/Tucker-Brianna.vit</measurements>",
                "<increments/>", "<draw name=\"Test1\">", "<calculation>", "<point id=\"1\" mx=\".1\" my=\".1\" name=\"A\" type=\"single\" x=\"90\" y=\"90\"/>",
                "<point angle=\"90\" basePoint=\"1\" id=\"2\" length=\"height*2\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"A1\" type=\"endLine\" typeLine=\"hair\"/>",
                "<point angle=\"180\" basePoint=\"1\" id=\"3\" length=\"height*2\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"A2\" type=\"endLine\" typeLine=\"hair\"/>",
                "<point id=\"4\" mx=\".1\" my=\".1\" name=\"B\" type=\"single\" x=\"20\" y=\"20\"/>"]
repeatLines2 = ["</calculation>","<modeling/>","<details/>","</draw>","</pattern>"]

#THE PROCESS:
# Find all the vertices of the shape
# Organize vertices into pairs (representing the features)
# Between each vertex - determine if it is a line or a curve    (HERE AFTER DAY1)
  #  UP IN AIR
    # CAN USE POLY FIT ON SECTION OF FILE BETWEEN THE VERTICES
    # CAN USE HOUGH TRANSFORM TO GET LINE, IF NOT LINE, SOMETHING ELSE
# Create a Directed A-Graph of the features
# Take the list of features (lines or curves) and map them to images
  # If it is a line:
    # Map the line too the proper (measurement, angle) image      (NEED TO BE HERE AFTER DAY2)
  # If it is a curve:
    # Figure out what type of curve it is (selection of 100)
    # Map the type of curve to its image representing that type
# Take the XML file and convert it to RSL
  # If it is a line the RSL is ("measurement-angle")
  # If it is a curve:
    # The xml get fed to a process to categorize what type of curve it is (selection of 100)
    # The RSL is output with the type of curve it is ("Type-#")    (NEED TO BE HERE AFTER DAY3)
#Run the images and the RSL through the model so it can train on all the image files and RSL files
#FIRST SECOND point of simpleInteractive (Probably needs to be restructured to calc lines first, then curves.)
#Fix control point calculation
#Up until now has been to create the data. Need to build out the learning to train features to give us XML
    #Create the full angle, distance mapping to control Points
    #create control point1, control poitn 2 mapping to image
    #be able to take image names and output their actual image to a file
    #Generate the RSL from XML (RSL can just be full lines as tokens)

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
            return(int(360 - abs(angle*57.2958)))

        Feature.__init__(self,x1,y1,x2,y2)
        self.angle = round(math.atan2(y2-y1, x2-x1),2)*-1
        self.degrees = calc_degrees(self.angle)
        self.distance = my_round(int(math.sqrt((x2-x1)**2 + (y2-y1)**2)),5)
        self.orderNum = order
        self.endpoints = (x1,y1),(x2,y2)
class Curve(Feature):

    orderNum = ...;

    def __init__(self, x1, y1, x2, y2, x3, y3, x4, y4, order):

        def my_round(x, roundNum):
            return round(x/roundNum)*roundNum

        def calc_degrees(angle):
            if angle > 0:
                return(int(angle*57.2958))
            return(int(360 - abs(angle*57.2958)))

        Feature.__init__(self,x1,y1,x4,y4)

        #second angle calc due to control points order
        self.angle1 = round(math.atan2(y2-y1, x2-x1),2)*-1; self.distance1 = my_round(int(math.sqrt((x2-x1)**2 + (y2-y1)**2)),5);
        self.angle2 = round(math.atan2(y3-y4, x3-x4),2)*-1; self.distance2 = my_round(int(math.sqrt((x4-x3)**2 + (y4-y3)**2)),5);
        self.degrees1 = calc_degrees(self.angle1); self.degrees2 = calc_degrees(self.angle2)
        self.orderNum = order;
        self.endpoints = (x1,y1),(x4,y4)

def is_close(x1, y1, x2, y2, d):
    distance = int(math.sqrt((x2-x1)**2 + (y2-y1)**2))
    return(distance<d)

def clean_Image(image):
    img = cv2.imread(image)
    height, width, channels = img.shape
    img_border =  cv2.copyMakeBorder(img,25,25,25,25,cv2.BORDER_CONSTANT,value=255)
    height, width, channels = img_border.shape
    new_image = np.ones((height,width,3), np.uint8)*255
    for i in range(0,height):
        for j in range(0,width):
            if np.any(img_border[i,j] > 4):
                img_border[i,j] = [255,255,255]
    gray = cv2.cvtColor(img_border,cv2.COLOR_BGR2GRAY)
    img_conv = util.invert(gray)
    kernel = np.ones((5,5),np.uint8)
    close = cv2.dilate(img_conv, kernel, iterations = 2)
    img_conv = util.invert(close)
    ret, thresh = cv2.threshold(img_conv, 100, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    contours_by_size = sorted(contour_sizes, key=lambda x: x[0])
    pattern_contour = contours_by_size[-2][1]
    cv2.drawContours(new_image, pattern_contour, -1, (0,0,0), 3)
    new_image = cv2.erode(new_image, kernel, iterations = 2)
    cv2.imwrite('/Users/theodoreseem/Desktop/subpixel.png',new_image)
    return new_image

def retrieveVertices(image):

    img = image
    height, width, channels = img.shape
    #kernel = np.ones((10,10),np.uint8)
    #erosion = cv2.erode(img,kernel,iterations = 1)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # find Harris corners
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,3,11,.1)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,100,255,0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, .01)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(10,10),(-1,-1),criteria)
    vertices = list(np.int0(corners))
    for v in vertices:
        if is_close(v[0], v[1], (width/2), (height/2), 20):
            vertices.remove(v)
    vertices = np.array(vertices)
    #print(vertices)
    for v in vertices:
        cv2.circle(img,(v[0],v[1]), 5, (255,0,0), 5)
    height, width, channels = img.shape
    #for i in range(0,height):
    #    for j in range(0,width):
    #        if np.any(img[i,j] != 255):
    #            img[i,j] = [255,0,0]
    cv2.imwrite('/Users/theodoreseem/Desktop/subpixel5.png',img)
    return vertices

def createFeatures(vertices):

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

def createSubFeatures(featureList, vertices, image):

    def collinear_exact(Points):
        Points.sort(key=lambda x: x[0])
        #print(Points)
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
    def findAssociatedFeature(contour):
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
    def findControlPoints(points):


        #clusters contours into center line
        #points = [(int((p1[0]+p2[0]+p3[0]+p4[0])/4),int((p1[1]+p2[1]+p3[1]+p4[1])/4)) for p1,p2,p3,p4 in zip(points[0::4], points[1::4], points[2::4], points[3::4])]

        xs = [x for x,y in points]
        ys = [y for x,y in points]

        xSample1 = points[0][0]; xSample2 = points[int(len(points)/2)][0] ; xSample3 = points[len(points)-1][0]
        if abs(xSample1-xSample2)<50 and abs(xSample1-xSample3)<50:
            print("vertical line")
            tmp = xs, ys
            ys = tmp[0]; xs = tmp[1]
            polynomial = np.poly1d(np.polyfit(xs, ys, 3))
            ys = [polynomial(x) for x in xs]
            tmp = xs, ys
            ys = tmp[0]; xs = tmp[1]
        else:
            polynomial = np.poly1d(np.polyfit(xs, ys, 3))
            ys = [polynomial(x) for x in xs]

        polys = [(int(x),int(y)) for x,y in zip(xs,ys)]  #NEED TO GET THIS POLYNOMIAL LINE TO FIT BETTER

        ts = np.arange(0.0, 1.0, 1.0/len(points))
        # make the summation functions for A (16 of them)
        A_fns = [None for i in range(16)]
        A_fns[0] = lambda time_list, data_list : sum([2*(t_i - 1)**6 for t_i, d_i in zip(time_list, data_list)])
        A_fns[1] = lambda time_list, data_list : sum([-6*t_i*(t_i - 1)**5 for t_i, d_i in zip(time_list, data_list)])
        A_fns[2] = lambda time_list, data_list : sum([6*t_i**2*(t_i - 1)**4 for t_i, d_i in zip(time_list, data_list)])
        A_fns[3] = lambda time_list, data_list : sum([-2*t_i**3*(t_i - 1)**3 for t_i, d_i in zip(time_list, data_list)])
        A_fns[4] = lambda time_list, data_list : sum([-6*t_i*(t_i - 1)**5 for t_i, d_i in zip(time_list, data_list)])
        A_fns[5] = lambda time_list, data_list : sum([18*t_i**2*(t_i - 1)**4 for t_i, d_i in zip(time_list, data_list)])
        A_fns[6] = lambda time_list, data_list : sum([-18*t_i**3*(t_i - 1)**3 for t_i, d_i in zip(time_list, data_list)])
        A_fns[7] = lambda time_list, data_list : sum([6*t_i**4*(t_i - 1)**2 for t_i, d_i in zip(time_list, data_list)])
        A_fns[8] = lambda time_list, data_list : sum([6*t_i**2*(t_i - 1)**4 for t_i, d_i in zip(time_list, data_list)])
        A_fns[9] = lambda time_list, data_list : sum([-18*t_i**3*(t_i - 1)**3 for t_i, d_i in zip(time_list, data_list)])
        A_fns[10] = lambda time_list, data_list : sum([18*t_i**4*(t_i - 1)**2 for t_i, d_i in zip(time_list, data_list)])
        A_fns[11] = lambda time_list, data_list : sum([-6*t_i**5*(t_i - 1) for t_i, d_i in zip(time_list, data_list)])
        A_fns[12] = lambda time_list, data_list : sum([-2*t_i**3*(t_i - 1)**3 for t_i, d_i in zip(time_list, data_list)])
        A_fns[13] = lambda time_list, data_list : sum([6*t_i**4*(t_i - 1)**2 for t_i, d_i in zip(time_list, data_list)])
        A_fns[14] = lambda time_list, data_list : sum([-6*t_i**5*(t_i - 1) for t_i, d_i in zip(time_list, data_list)])
        A_fns[15] = lambda time_list, data_list : sum([2*t_i**6 for t_i, d_i in zip(time_list, data_list)])

        # make the summation functions for b (4 of them)
        b_fns = [None for i in range(4)]
        b_fns[0] = lambda time_list, data_list : -1.0 * sum([2*d_i*(t_i - 1)**3 for t_i, d_i in zip(time_list, data_list)])
        b_fns[1] = lambda time_list, data_list : -1.0 * sum([-6*d_i*t_i*(t_i - 1)**2 for t_i, d_i in zip(time_list, data_list)])
        b_fns[2] = lambda time_list, data_list : -1.0 * sum([6*d_i*t_i**2*(t_i - 1) for t_i, d_i in zip(time_list, data_list)])
        b_fns[3] = lambda time_list, data_list : -1.0 * sum([-2*d_i*t_i**3 for t_i, d_i in zip(time_list, data_list)])

        def solve_for_cs(time_series, data_series):
            """
            Take an input series of t_i values and the corresponding d_i values,
            compute the summation values that should go into the matrices and
            solve for the 4 unknown variables.

            Parameters: time_series -- t_i in increasing values
                        data_series -- d_i corresponding to each t_i

            Returns: solution -- matrix containing the 4 solutions from solving the linear equations
            """
            # compute the data we will put into matrix A
            A_values = []
            for fn in A_fns:
                A_values.append(fn(time_series, data_series))
            # fill the A matrix with data
            A_numerical = Matrix(4,4, A_values)

            # compute the data we will put into the b vector
            b_values = []
            for fn in b_fns:
                b_values.append(fn(time_series, data_series))
            # fill the b vector with data
            b_numerical = Matrix(4,1, b_values)

            #print(A_numerical, b_numerical)

            # solve for the unknowns in vector x
            x_numerical = A_numerical.inv() * b_numerical

            return x_numerical

        # solve for the best fit in the x dimension
        x_solutions = solve_for_cs(time_series=ts, data_series=xs)
        # solve for the best fit in the y dimension
        y_solutions = solve_for_cs(time_series=ts, data_series=ys)

        # place the solutions into control points
        best_fit_control_pts = [(int(x),int(y)) for x,y in zip(x_solutions, y_solutions)]

        return best_fit_control_pts, polys

    feaList = []; imageList = []
    img = image
    height, width, channels = img.shape
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_conv = util.invert(gray)
    img_conv = cv2.erode(img_conv, np.ones((3,3),np.uint8), iterations = 2)
    for v in vertices:
        cv2.circle(img_conv,(v[0],v[1]), 15, (0,0,0), -1)
    ret, thresh = cv2.threshold(img_conv, 100, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        c_arr = np.vstack(c).squeeze()
        c_tup = list(map(tuple, c_arr))
        isLine = collinear_approx(c_tup)
        xList = [p[0][0] for p in c]
        yList = [p[0][1] for p in c]
        if isLine:
           feature = findAssociatedFeature(c_tup)
           lineObj = Line(feature.x1, feature.y1, feature.x2, feature.y2, feature.orderNum)
           feaList.append(lineObj)
           #print("Line: (ang, dis)", lineObj.degrees, lineObj.distance)
           lineMeasure = lExt_mapping[str(lineObj.distance)],int(abs(lineObj.degrees))
           lineImage = lineImg_mapping[str(lineMeasure)]
           #print("line mapped to: ", lineImage)
           imageList.append(lineImage)
        else:

            #PASSING THE CONTROL POINTS AND THE POINTS OF THE POLYNOMIAL FIT
            #The curve end points aren't correct and then need to re-align them into proper DAG order
            #Then check that the Seamly2d VERSION RENDERS SOMEWHAT CLOSE TO THE REAL IMAGE

            cps, polys = findControlPoints(c_tup)
            feature = findAssociatedFeature(c_tup)
            cv2.line(image, cps[0], cps[1] , (0,255,0), 5)
            cv2.line(image, cps[2], cps[3], (0,255,0), 5)
            for p in polys: cv2.circle(image,p, 1, (0,255,0), -1)
            for p in c_tup: cv2.circle(image,p, 1, (0,0,255), -1)
            curveObj = Curve(cps[0][0], cps[0][1], cps[1][0], cps[1][1], cps[2][0],cps[2][1], cps[3][0], cps[3][1], feature.orderNum)
            #print("Curve: (a1,d1,a2,d2)", curveObj.degrees1, curveObj.distance1, curveObj.degrees2, curveObj.distance2)
            #print("Curve: (x1,y1,x2,y2)", curveObj.x1, curveObj.y1, curveObj.x2, curveObj.y2)
            feaList.append(curveObj)
            curveMeasure1 = cExt_mapping[str(curveObj.degrees1)+", "+str(curveObj.distance1)]
            curveMeasure2 = cExt_mapping[str(curveObj.degrees2)+", "+str(curveObj.distance2)]
            curveImage = curveImg_mapping[str(curveMeasure1)+", "+str(curveMeasure2)]
            #print("curve mapped to: ", curveImage)
            imageList.append(curveImage)

    cv2.imwrite('/Users/theodoreseem/Desktop/controlpoints.png',image)
    feaList.sort(key = lambda x: x.orderNum)
    return feaList, imageList, height

def compileXML(featList, output_file, height):
    editableOutput = open(output_file, "w") #subbed for print for now

    #PROBABLY NEED TO ALIGN THE STARTING POINT (4) TO THE INITIAL X Y VALUES FOR THE STARTING FEATURE

    shrink = 50
    curve_offset = 50
    x_off = 0 # featList[0].x1/50;
    y_off = 0 #featList[0].y1/50
    startPoint = 4

    for f in featList:
        f.x1 = (f.x1/shrink)-x_off; f.x2 = (f.x2/shrink)-x_off; f.y1 = (f.y1/shrink)-y_off; f.y2 = (f.y2/shrink)-y_off
        if type(f) is Line:
            f.distance = f.distance/shrink
        else:
            f.distance1 = f.distance1/shrink
            f.distance2 = f.distance2/shrink

    for line in repeatLines:
        print(line + "\n")

    for counter, feat in enumerate(featList[:-1]):
        if type(feat) is Line:
            print("<point angle=\"%d\" basePoint=\"%d\" id=\"%d\" length=\"%d\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"B1\" type=\"endLine\" typeLine=\"hair\"/>\n" %(feat.degrees, counter+startPoint, counter+startPoint+1, feat.distance))
        else:
            print("<point id=\"%d\" mx=\"0\" my=\"0\" name=\"B\" type=\"single\" x=\"%f\" y=\"%f\"/>\n" % (counter+startPoint+1, feat.x2, feat.y2))
            print("<spline angle1=\"%d\" angle2=\"%d\" color=\"black\" id=\"%d\" length1=\"%d\" length2=\"%d\" penStyle=\"hair\" point1=\"%d\" point4=\"%d\" type=\"simpleInteractive\"/>\n" % (feat.degrees1, feat.degrees2, counter+curve_offset, feat.distance1, feat.distance2, counter+startPoint, counter+startPoint+1))
    if type(featList[-1]) is Line:
        print("<point angle=\"%d\" basePoint=\"%d\" id=\"%d\" length=\"%d\" lineColor=\"darkRed\" mx=\".1\" my=\".1\" name=\"B1\" type=\"endLine\" typeLine=\"hair\"/>\n" %(featList[-1].degrees, counter+startPoint, startPoint, featList[-1].distance))
    else:
        print("<spline angle1=\"%d\" angle2=\"%d\" color=\"darkRed\" id=\"%d\" length1=\"%d\" length2=\"%d\" penStyle=\"hair\" point1=\"%d\" point4=\"%d\" type=\"simpleInteractive\"/>\n" % (featList[-1].degrees1, featList[-1].degrees2, curve_offset+12, featList[-1].distance1, featList[-1].distance2, len(featList)-1+startPoint, startPoint))

    for line in repeatLines2:
        print(line + "\n")

    return "foo"

def buildImageRepresentation(imageList):


    #images = [cv2.imread('/Users/theodoreseem/res.Network/Hasy-images/' + image) for image in imageList]
    files = ['/Users/theodoreseem/res.Network/Hasy-images/' + image for image in imageList]

    result = Image.new("RGB", (800, 800))

    for index, file in enumerate(files):
        path = os.path.expanduser(file)
        img = Image.open(path)
        img.thumbnail((400, 400), Image.ANTIALIAS)
        x = index // 2 * 400
        y = index % 2 * 400
        w, h = img.size
        print('pos {0},{1} size {2},{3}'.format(x, y, w, h))
        result.paste(img, (x, y, x + w, y + h))

    result.save(os.path.expanduser('Users/theodoreseem/Desktop/imageTEST.jpg'))



if __name__ == "__main__":

    argv = sys.argv[1:]
    arg_length = len(argv)
    if arg_length != 0:
        input_Image = argv[0]
        output_file = argv[1]
    else:
        print("Error: not enough argument supplied:")
        print("compiler.py <path> <file name>")
        exit(0)

    print(input_Image)
    with open(lineImage_mapping) as data_file:
        lineImg_mapping = json.load(data_file); lineImg_mapping = defaultdict(lambda: "NO IMG", lineImg_mapping)
    with open(curveImage_mapping) as data_file:
        curveImg_mapping = json.load(data_file); curveImg_mapping = defaultdict(lambda: "NO IMG", curveImg_mapping)
    with open(lineExt_mapping) as ext_file:
        lExt_mapping = json.load(ext_file); lExt_mapping = defaultdict(lambda: "NO MEASURE", lExt_mapping)
    with open(curveExt_mapping) as ext_file:
        cExt_mapping = json.load(ext_file); cExt_mapping = defaultdict(lambda: "NO MEASURE", cExt_mapping)

    image_cleaned = clean_Image(input_Image)
    image_vertices = retrieveVertices(image_cleaned)
    fList = createFeatures(image_vertices)
    subFeatureList, imageList, height = createSubFeatures(fList, image_vertices, image_cleaned)
    for f in subFeatureList:
        print("Curve: (x1,y1,x2,y2)", f.x1, f.y1, f.x2, f.y2)
    #compiledXML = compileXML(subFeatureList, output_file, height)
    #buildImageRepresentation(imageList) #CANT DO THIS UNTIL THE DICTIONARY IS BUILT OUT
