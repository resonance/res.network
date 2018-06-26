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
import copy
from PIL import Image
import statistics
import random
np.set_printoptions(threshold=np.nan)


from sympy import Matrix

## TODO: Fix the placement of Corners (use overlap of dst and curve and then find centroid), need notches
## TODO: Fix the bezier curve calculation (Second point seems to be endpoint)
## TODO: Replace the carcoded paths thorughout the file


## TODO: Clean up the code in both res.Network accounts (GitHub, AWS)

degree_round = 10
distance_round = 25
conversionShrinkage = 50

master_Directory = '/Users/theodoreseem/res.Network/Curve_Integration/'

lineImage_mapping = '/Users/theodoreseem/res.Network/Curve_Integration/dictionaries/Line-Image.json'
featureExt_mapping = '/Users/theodoreseem/res.Network/Curve_Integration/dictionaries/Feature-Extraction.json'
curveImage_mapping = '/Users/theodoreseem/res.Network/Curve_Integration/dictionaries/Curve-Image.json'

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
        self.x1 = x1; self.y1 = y1; self.x2 = x2; self.y2 = y2

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
        self.x1 = x1; self.y1 = y1; self.x2 = x2; self.y2 = y2; self.x3 = x3; self.y3 = y3

        def my_round(x, roundNum):
            return round(x/roundNum)*roundNum

        def calc_degrees(angle):
            if angle > 0:
                return(int(angle*57.2958))
            return(int(360 - abs(angle*57.2958)))

        Feature.__init__(self,x1,y1,x3,y3)

        #second angle calc due to control points order
        self.angle = round(math.atan2(y2-y1, x2-x1),2)*-1
        self.distance = my_round(int(math.sqrt((x2-x1)**2 + (y2-y1)**2)),5)
        self.degrees = calc_degrees(self.angle);
        self.orderNum = order;
        self.endpoints = (x1,y1),(x3,y3)

        #Depricated from a time when calculating cubic curves and nto quadratic curves
            #self.distance1 = my_round(int(math.sqrt((x2-x1)**2 + (y2-y1)**2)),5); self.distance2 = my_round(int(math.sqrt((x2-x3)**2 + (y2-y3)**2)),5);
            #self.angle1 = round(math.atan2(y2-y1, x2-x1),2)*-1; self.angle2 = round(math.atan2(y2-y3, x2-x3),2)*-1
            #self.degrees1 = calc_degrees(self.angle1); self.degrees2 = calc_degrees(self.angle2)

class Point:

    #Used in the partitioning of features
    slopetToAverage = ...;

    def __init__(self, x, y):
        self.x = x; self.y = y;

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
        if dist>4: return False
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

def createDAG(vertices, img):

    height, width, channels = img.shape

    def get_dist(x1,y1,x2,y2):
        return int(math.sqrt((x2-x1)**2 + (y2-y1)**2))
    def clockwiseangle_and_distance(point):
        origin = [height/2, width/2]
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


    '''This is the fucntion to take the clustered output from previous line and organize into arrays of clustered points'''
    '''Storing in two dimensional array where the first position of each second dimension holds all the points coordinates'''
    '''Said differently componentArray[0][0] holds coordinates of first feature, componentArray[1][0] the second feature, componentArray[2][0] the third...'''
    filledFeatures = []

    for component in range(1,len(lab_stats)):
        indices = np.where(img_labeled == component)
        indices = tuple(zip(*indices))
        indices = [t[::-1] for t in indices]

        feature = findAssociatedFeature(indices, DAG)
        indexPoints = []
        for i in indices:
            indexPoints.append(Point(i[0],i[1]))
        feature.points = indexPoints
        filledFeatures.append(feature)

    filledFeatures.sort(key=lambda x: DAG.index(x))

    return filledFeatures

def divideFeatures(features):



    #Filter the points
    #Find the center point
    #Divide filter points into two groups
    #create two new features
    #replace previous feature with the two new features in the DAG
    #run the clockwise function on the new features endpoints to determine first or last

    def iswWithin(bx1, bx2, by1, by2, p):
        if bx1<p[0]<bx2 and by1<p[1]<by2:
            return True
        return False
    def filter_points(pointsObjs, step_size = 75):

        points = []
        for p in pointsObjs:
            points.append((p.x,p.y))

        '''potential TODO Could get facier here and remove all points within x of any other point. This would help with vertical line deteciton later on'''
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

        filteredPointObjs = []
        for p in filtered_points:
            filteredPointObjs.append(Point(p[0],p[1]))

        return filteredPointObjs
    def closestTo(anchor, points):
        closestDist = int(math.sqrt((points[0][0]-anchor[0])**2 + (points[0][1]-anchor[1])**2))
        closestPoint = points[0]
        for p in points:
            distToAnchor = int(math.sqrt((p[0]-anchor[0])**2 + (p[1]-anchor[1])**2))
            if distToAnchor < closestDist:
                closestDist = distToAnchor
                closestPoint = p
        return closestPoint

    ''' Iterate through each feature and find the filtered points of it's contour points. This is done by evaluating 50x50 area dimensions of that feature area
        and averaging all points which fall into that 50x50 grid location
        If a feature's length is less than some (200) pixels in length, the sub pixels are populated as the endpoints (because no need to divide feature further)
    '''

    newFeatureSet = []

    ##TODO: Fix DAG with new features


    for f in features:
        if f.distance > 200:
            f.filteredPoints  = filter_points(f.points);
            f.filteredPoints.extend([Point(f.endpoints[0][0], f.endpoints[0][1]),Point(f.endpoints[1][0], f.endpoints[1][1])])
            AveragePoint = filter_points(f.points, 100000)[0]
            indices = [(p.x,p.y) for p in f.filteredPoints]
            if not collinear_approx(indices):   #if it is a curve
                for p in f.filteredPoints:
                    p.slopetToAverage = round(math.atan2(AveragePoint.y-p.y, AveragePoint.x-p.x),2)*-1
                sortedFiltered = sorted(f.filteredPoints, key=lambda p: p.slopetToAverage)
                firstHalfPoints = sortedFiltered[:len(sortedFiltered)//2]; secondHalfPoints = sortedFiltered[len(sortedFiltered)//2:];
                feature_1 = Feature(firstHalfPoints[0].x, firstHalfPoints[0].y,firstHalfPoints[len(firstHalfPoints)-1].x, firstHalfPoints[len(firstHalfPoints)-1].y)
                feature_1.filteredPoints = firstHalfPoints
                feature_1.orderNum = f.orderNum
                feature_2 = Feature_1 = Feature(secondHalfPoints[0].x, secondHalfPoints[0].y,secondHalfPoints[len(secondHalfPoints)-1].x, secondHalfPoints[len(secondHalfPoints)-1].y)
                feature_2.filteredPoints = secondHalfPoints
                feature_2.orderNum = f.orderNum
                newFeatureSet.append(feature_1); newFeatureSet.append(feature_2)
            #closest_toAvg = closestTo(AveragePoint, f.filteredPoints)
        else:
            f.filteredPoints = filter_points(f.points)
            newFeatureSet.append(copy.deepcopy(f))

    return newFeatureSet

def createSubFeatures(features):


    subFeatures = []
    is_vertical = False
    def cubicControlPoints(points):

        def polyfit_with_fixed_points(n, x, y, xf, yf) :
            mat = np.empty((n + 1 + len(xf),) * 2)
            vec = np.empty((n + 1 + len(xf),))
            x_n = x**np.arange(2 * n + 1)[:, None]
            yx_n = np.sum(x_n[:n + 1] * y, axis=1)
            x_n = np.sum(x_n, axis=1)
            idx = np.arange(n + 1) + np.arange(n + 1)[:, None]
            mat[:n + 1, :n + 1] = np.take(x_n, idx)
            xf_n = xf**np.arange(n + 1)[:, None]
            mat[:n + 1, n + 1:] = xf_n / 2
            mat[n + 1:, :n + 1] = xf_n.T
            mat[n + 1:, n + 1:] = 0
            vec[:n + 1] = yx_n
            vec[n + 1:] = yf
            params = np.linalg.solve(mat, vec)
            return params[:n + 1][::-1]

        #clusters contours into center line
        points = [(int((p1[0]+p2[0]+p3[0]+p4[0])/4),int((p1[1]+p2[1]+p3[1]+p4[1])/4)) for p1,p2,p3,p4 in zip(points[0::4], points[1::4], points[2::4], points[3::4])]

        xs = [x for x,y in points]
        ys = [y for x,y in points]

        #xSample1 = points[0][0]; xSample2 = points[int(len(points)/2)][0] ; xSample3 = points[len(points)-1][0]
        #if abs(xSample1-xSample2)<50 and abs(xSample1-xSample3)<50:
        #    print("vertical line")
        #    tmp = xs, ys
        #    ys = tmp[0]; xs = tmp[1]
        #    #polynomial = np.poly1d(np.polyfit(xs, ys, 3))
        #    polynomial = np.poly1d(polyfit_with_fixed_points(4, xs , ys, [xs[0],xs[int(len(xs)/2)], xs[len(xs)-1]], [ys[0],ys[int(len(ys)/2)], ys[len(ys)-1]]))
        #    ys = [polynomial(x) for x in xs]
        #    tmp = xs, ys
        #    ys = tmp[0]; xs = tmp[1]
        #else:
        #    #polynomial = np.poly1d(np.polyfit(xs, ys, 3))
        #    polynomial = np.poly1d(polyfit_with_fixed_points(4, xs , ys, [xs[0],xs[int(len(xs)/2)], xs[len(xs)-1]], [ys[0],ys[int(len(ys)/2)], ys[len(ys)-1]]))
        #    ys = [polynomial(x) for x in xs]

        #polynomial = np.poly1d(polyfit_with_fixed_points(3, xs , ys, [xs[0],xs[int(len(xs)/2)], xs[len(xs)-1]], [ys[0],ys[int(len(ys)/2)], ys[len(ys)-1]]))



        #polys = [(int(x),int(y)) for x,y in zip(xs,ys)]  #NEED TO GET THIS POLYNOMIAL LINE TO FIT BETTER

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

        return best_fit_control_pts
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
    def needsTransform(points):
        #return abs(points[0][0]-points[int(len(points)/2)][0])<500 and abs(points[0][0]-points[-1][0])<500 #Assumption
        for p1 in points:
            for p2 in points:
                if abs(p1.x-p2.x) < 4 and p2 is not p1:
                    return True
        return False

    new_image = np.ones((2000,2000,3), np.uint8)*255

    for f in features:
        indices = [(p.x,p.y) for p in f.filteredPoints]
        if collinear_approx(indices):
            lineObj = Line(f.x1, f.y1, f.x2, f.y2, f.orderNum)
            subFeatures.append(lineObj)
        else:
            for p in f.filteredPoints:
                cv2.circle(new_image, (p.x, p.y), 5, (255,0,0), -1)
            if needsTransform(f.filteredPoints):
                print(f.endpoints)
                is_vertical = True
                xs = [p.x for p in f.filteredPoints]; ys = [p.y for p in f.filteredPoints]
                tmp = xs, ys
                ys = tmp[0]; xs = tmp[1]
            else:
                xs = [p.x for p in f.filteredPoints]; ys = [p.y for p in f.filteredPoints]

            polynomial = np.poly1d(np.polyfit(xs, ys, 4))
            ys = [int(polynomial(x)) for x in xs]
            if is_vertical:
                tmp = xs, ys
                ys = tmp[0]; xs = tmp[1]
                is_vertical = False
            polyPoints = zip(xs, ys)
            for p in polyPoints:
                cv2.circle(new_image, (p[0], p[1]), 5, (0,255,0), -1)

            x_cp, y_cp = calc_quadratic_bezier(polynomial,f.x1,f.x2)
            cv2.line(new_image, (f.x1, f.y1), (x_cp, y_cp), (0,0,255), 5)


            curveObj = Curve(f.x1, f.y1, x_cp, y_cp, f.x2, f.y2, f.orderNum)
            subFeatures.append(curveObj)

    assignment = random.randint(0, 1000)
    cv2.imwrite('/Users/theodoreseem/res.Network/Curve_Integration/test/test' + str(assignment) + '.png', new_image)

    return subFeatures

def createImageList(subFeatures):

    imageList = []
    for subF in subFeatures:
        if type(subF) == Line:
            featureMeasure = feat_ext_map[str(floor_round(subF.distance,distance_round))],int(abs(floor_round(subF.degrees,degree_round)))
            lineImage = line_Image_map[str(featureMeasure)]
            #print(featureMeasure)
            imageList.append(lineImage)
        else:
            featureMeasure = feat_ext_map[str(floor_round(subF.distance,distance_round))],int(abs(floor_round(subF.degrees,degree_round)))
            curveImage = curve_Image_map[str(featureMeasure)]
            #print(featureMeasure)
            imageList.append(curveImage)

    return imageList

def createRSL(subFeatures, outputRSL):

    openOutput = open(outputRSL, "w")

    openOutput.write("first-point\n")
    openOutput.write("frame-one\n")
    openOutput.write("frame-two\n")
    openOutput.write("second-point\n")

    for subf in subFeatures:
        wordOutput = "RSL-%s-%s" %(str(floor_round(float(subf.degrees),degree_round)),str(floor_round(float(subf.distance)*conversionShrinkage, distance_round)))
        openOutput.write(wordOutput+ '\n')

    openOutput.write("end-line")
    openOutput.close()

def createRIL(imageList, output_RIL):

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

def cleanAndBuild(file, imageStore):

    input_Image = master_Directory + imageStore + "/" + file + ".JPEG"
    output_XML =  master_Directory + "xml_outputs/" + file + ".val"
    output_RSL =  master_Directory + "rsl_outputs/" + file + ".gui"
    output_RIL =  master_Directory + "ril_outputs/" + file + ".png"
    output_cleaned =  master_Directory + "cleaned_images/" + file + ".png"

    imageCleaned = clean_Image(input_Image)
    image_vertices = retrieveVertices(imageCleaned)
    DAG = createDAG(image_vertices, imageCleaned)
    initialFeatures = extract_initial_contours(DAG, imageCleaned)
    refinedFeatures = divideFeatures(initialFeatures)
    subFeatures = createSubFeatures(refinedFeatures)
    imageList = createImageList(subFeatures)
    if 'NO IMG' not in imageList:
        createRIL(imageList, output_RIL)
        createRSL(subFeatures, output_RSL)
    else:
        print("Skipping RIL Creation")




#    height, width, channels = imageCleaned.shape
#    new_image = np.ones((height,width,3), np.uint8)*255
#    for f in range(len(initialFeatures)):
#        featurePoints = initialFeatures[f][0]
#        refinedPoints = refinedFeatures[f][0]
#        delPoints = delineatedFeatures[f][0]

#        for p in featurePoints:
#            cv2.circle(new_image, (p[0], p[1]), 1, (255,0,0), -1)
#        for m in refinedPoints:
#            cv2.circle(new_image, (m[0], m[1]), 1, (0,255,0), -1)
#        for m in delPoints:
#            cv2.circle(new_image, (m[0], m[1]), 10, (0,0,255), -1)
#    assignment = random.randint(0, 1000)
#    cv2.imwrite('/Users/theodoreseem/res.Network/Curve_Integration/test/test' + str(assignment) + '.png', new_image)

#NOT BEING USED
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
    if arg_length == 2:
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
