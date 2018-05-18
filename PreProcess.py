import sys
import cv2
import numpy as np
import math
import json
from skimage import util
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
from scipy.spatial import distance as dist
from sympy import Matrix
import scipy.ndimage, scipy.interpolate



lineImage_mapping = '/Users/theodoreseem/Desktop/Emoji2Code/dictionaries/Line-Image_mapping.json'
curveImage_mapping = '/Users/theodoreseem/Desktop/Emoji2Code/dictionaries/Curve-Image_mapping.json'
lineExt_mapping = '/Users/theodoreseem/Desktop/Emoji2Code/dictionaries/lineExtraction_mapping.json'
curveExt_mapping = '/Users/theodoreseem/Desktop/Emoji2Code/dictionaries/curveExtraction_mapping.json'

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

def retrieveVertices(image):
    img = cv2.imread(image)
    height, width, channels = img.shape
    #kernel = np.ones((10,10),np.uint8)
    #erosion = cv2.erode(img,kernel,iterations = 1)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # find Harris corners
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.02)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.3*dst.max(),255,0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, .001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(10,10),(-1,-1),criteria)
    vertices = list(np.int0(corners))
    for v in vertices:
        if is_close(v[0], v[1], (width/2), (height/2), 20):
            vertices.remove(v)
    vertices = np.array(vertices)
    #print(vertices)
    #for v in vertices:
    #    cv2.circle(img,(v[0],v[1]), 10, (0,255,0), 10)
    #height, width, channels = img.shape
    #for i in range(0,height):
    #    for j in range(0,width):
    #        if np.any(img[i,j] != 255):
    #            img[i,j] = [255,0,0]
    #cv2.imwrite('subpixel5.png',img)
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
    feat = Feature(ccWise[len(ccWise)-1][0], ccWise[len(ccWise)-1][1],ccWise[0][0], ccWise[0][1])
    featureList.append(feat)
    for count, f in enumerate(featureList):
        f.orderNum = count
    return featureList

def createSubFeatures(featureList, vertices, image):

    def collinear_exact(Points):
        Points.sort(key=lambda x: x[0])
        print(Points)
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
            if dist>6: return False
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
        print("Internal Error: measurement not found")
        exit(0)
    def findControlPoints(points, xs, ys):

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

            # solve for the unknowns in vector x
            x_numerical = A_numerical.inv() * b_numerical

            return x_numerical

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

        # solve for the best fit in the x dimension
        x_solutions = solve_for_cs(time_series=ts, data_series=xs)
        # solve for the best fit in the y dimension
        y_solutions = solve_for_cs(time_series=ts, data_series=ys)

        # place the solutions into control points
        best_fit_control_pts = [(int(x),int(y)) for x,y in zip(x_solutions, y_solutions)]

        return best_fit_control_pts

    test = 1
    feaList = []; imageList = []
    img = cv2.imread(image)
    height, width, channels = img.shape
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_conv = util.invert(gray)
    for v in vertices:
        cv2.circle(img_conv,(v[0],v[1]), 15, (0,0,0), -1)
    kernel = np.ones((2,2),np.uint8)
    close = cv2.dilate(img_conv, kernel, iterations = 2)
    ret, thresh = cv2.threshold(close, 100, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        #cv2.drawContours(thresh, c, -1, (0,0,0), 50)
        #cv2.imwrite('subpixel5.png',thresh)
        c_arr = np.vstack(c).squeeze()
        c_tup = list(map(tuple, c_arr))
        isLine = collinear_approx(c_tup)
        xList = [p[0][0] for p in c]
        #print("x",xList)
        yList = [p[0][1] for p in c]
        #yList = [height-y for y in yList]
        #print("y", yList)
        if isLine:
           feature = findAssociatedFeature(c_tup)
           lineObj = Line(feature.x1, feature.y1, feature.x2, feature.y2, feature.orderNum)
           feaList.append(lineObj)
           print("Line: (ang, dis)", lineObj.degrees, lineObj.distance)

           lineMeasure = lExt_mapping[str(lineObj.distance)],int(abs(lineObj.degrees))
           lineImage = lineImg_mapping[str(lineMeasure)]
           #print("line mapped to: ", lineImage)
           imageList.append(lineImage)
        else:
            cps = findControlPoints(c_tup, xList, yList)
            feature = findAssociatedFeature(c_tup)
            #print("Curve: (cntrl pts) ", cps) #Need to check the orientation of the Y axis for the control points
            if test == 1:
                for p in cps:
                    cv2.circle(thresh,(p[0],p[1]), 4, (255,255,255), 1)
                test = 0
            else:
                for p in cps:
                    cv2.circle(thresh,(p[0],p[1]), 3, (255,255,255), -1)
            cv2.imwrite('subpixel5.png',thresh)
            curveObj = Curve(cps[0][0], cps[0][1], cps[1][0], cps[1][1], cps[2][0],cps[2][1], cps[3][0], cps[3][1], feature.orderNum)
            print("Curve: (a1,d1,a2,d2)", curveObj.degrees1, curveObj.degrees2, curveObj.distance1, curveObj.distance2)
            feaList.append(curveObj)
            curveMeasure1 = cExt_mapping[str(curveObj.degrees1)+", "+str(curveObj.distance1)]
            curveMeasure2 = cExt_mapping[str(curveObj.degrees2)+", "+str(curveObj.distance2)]
            curveImage = curveImg_mapping[str(curveMeasure1)+", "+str(curveMeasure2)]
            #print("curve mapped to: ", curveImage)
            imageList.append(curveImage)

    feaList.sort(key = lambda x: x.orderNum)
    return feaList, imageList

def compileXML(featList, output_file):
    editableOutput = open(output_file, "w") #subbed for print for now

    for line in repeatLines:
        print(line + "\n")

    for counter, feat in enumerate(featList):
        if type(feat) is Line:
            print("<point angle=\"%d\" basePoint=\"%d\" id=\"%d\" length=\"%d\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"B1\" type=\"endLine\" typeLine=\"hair\"/>\n" %(feat.degrees, counter+4, counter+5, feat.distance))
        else:
            print("<spline angle1=\"%d\" angle2=\"%d\" color=\"black\" id=\"%d\" length1=\"%d\" length2=\"%d\" penStyle=\"hair\" point1=\"%d\" point4=\"%d\" type=\"simpleInteractive\"/>\n" % (feat.degrees1, feat.degrees2, counter+5, feat.distance1, feat.distance2, counter+4, counter+6))
    for line in repeatLines2:
        print(line + "\n")

    return "foo"


#    def createSubFeatures_old(featureList, vertices, image):
#        img = cv2.imread(image)
#        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#        img = util.invert(gray)
#        pad = 10
#        for f in featureList[7:8]:
#            if f.y1<f.y2 and f.x1<f.x2:
#                crop = img[f.y1-pad:f.y2+pad, f.x1-pad:f.x2+pad]
#            elif f.y1<f.y2 and f.x2<f.x1:
#                crop = img[f.y1-pad:f.y2+pad, f.x2-pad:f.x1+pad]
#            elif f.y2<f.y1 and f.x2<f.x1:
#                crop = img[f.y2-pad:f.y1+pad, f.x2-pad:f.x1+pad]
#            else:
#                crop = img[f.y2-pad:f.y1+pad, f.x1-pad:f.x2+pad]
#            kernel = np.ones((5,5),np.uint8)
#            close = cv2.dilate(crop, kernel, iterations = 1)
#            ret, thresh = cv2.threshold(close, 127, 255, 0)
#            im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#            xList = [x[0][0] for x in np.concatenate(contours)]
#            yList = [x[0][1] for x in np.concatenate(contours)]
        #    if line:
                #get angle and distance and map to image
        #    if curve:
                #get coefs integers and distance between edpoints and map to image
        #        coefs = np.polyfit(xList, yList, 5)
        #        ffit = np.polyval(coefs, xList)
        #        curveName = fitToPredefined()

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

    with open(lineImage_mapping) as data_file:
        lineImg_mapping = json.load(data_file); lineImg_mapping = defaultdict(lambda: "NO IMG", lineImg_mapping)
    with open(curveImage_mapping) as data_file:
        curveImg_mapping = json.load(data_file); curveImg_mapping = defaultdict(lambda: "NO IMG", curveImg_mapping)
    with open(lineExt_mapping) as ext_file:
        lExt_mapping = json.load(ext_file); lExt_mapping = defaultdict(lambda: "NO MEASURE", lExt_mapping)
    with open(curveExt_mapping) as ext_file:
        cExt_mapping = json.load(ext_file); cExt_mapping = defaultdict(lambda: "NO MEASURE", cExt_mapping)

    image_vertices = retrieveVertices(input_Image)
    fList = createFeatures(image_vertices)
    subFeatureList, imageList = createSubFeatures(fList, image_vertices, input_Image)
    compiledXML = compileXML(subFeatureList, output_file)
