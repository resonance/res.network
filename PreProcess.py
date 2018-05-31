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
import operator
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from sklearn.neighbors import NearestNeighbors
from scipy import linalg
from sklearn import mixture

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from scipy.interpolate import splprep, splev

numpy.set_printoptions(threshold=numpy.nan)

DEBUG_FLAG = True

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

def loadDictionaries():
    with open(lineImage_mapping) as data_file:
        lineImg_mapping = json.load(data_file); lineImg_mapping = defaultdict(lambda: "NO IMG", lineImg_mapping)
    with open(curveImage_mapping) as data_file:
        curveImg_mapping = json.load(data_file); curveImg_mapping = defaultdict(lambda: "NO IMG", curveImg_mapping)
    with open(lineExt_mapping) as ext_file:
        lExt_mapping = json.load(ext_file); lExt_mapping = defaultdict(lambda: "NO MEASURE", lExt_mapping)
    with open(curveExt_mapping) as ext_file:
        cExt_mapping = json.load(ext_file); cExt_mapping = defaultdict(lambda: "NO MEASURE", cExt_mapping)
    return lineImg_mapping, curveImg_mapping, lExt_mapping, cExt_mapping

def is_close(x1, y1, x2, y2, d):
    distance = int(math.sqrt((x2-x1)**2 + (y2-y1)**2))
    return(distance<d)

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
    cv2.drawContours(new_image, pattern_contour, -1, (0,0,0), 3)
    new_image = cv2.erode(new_image, kernel, iterations = 2)
    return new_image

def retrieveVertices(img):


    height, width, channels = img.shape
    erosion = cv2.dilate(img,np.ones((8,8),np.uint8),iterations = 1)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # find Harris corners
    dst = cv2.cornerHarris(np.float32(gray),3,11,.1)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,127,255,cv2.THRESH_BINARY)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst, connectivity=4)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, .01)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(10,10),(-1,-1),criteria)
    vertices = list(np.int0(corners))
    centerVertex = [v for v in vertices if is_close(v[0], v[1], (width/2), (height/2), 10)][0]
    vertices.remove(centerVertex)

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

def extract_initial_contours(vertices, img):

    height, width, channels = img.shape
    erosion = cv2.dilate(img,np.ones((8,8),np.uint8),iterations = 1)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_conv = util.invert(gray)
    img_conv = cv2.erode(img_conv, np.ones((3,3),np.uint8), iterations = 2)
    ret, thresh = cv2.threshold(img_conv, 100, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contourArray = np.vstack(contours[1]).squeeze()
    contourPoints = list(map(tuple, contourArray))

    def isCloseToVert(point):
        for v in vertices:
            if is_close(point[0], point[1], v[0], v[1], 5):
                return True
        return False

    points_without_vertices = [p for p in contourPoints if not isCloseToVert(p)]

    nonVertex_image = np.ones([height, width, 3],np.uint8)*255
    for p in points_without_vertices:
        cv2.circle(nonVertex_image, (p[0], p[1]), 1, (0,0,0), -1)

    erosion = cv2.erode(nonVertex_image,np.ones((2,2),np.uint8),iterations = 1)
    graynonVert = cv2.cvtColor(erosion,cv2.COLOR_BGR2GRAY)
    threshnonVert = cv2.threshold(graynonVert, 127, 255, cv2.THRESH_BINARY)[1]
    inverted = 255-threshnonVert

    img_bin = np.array(inverted)

    n_labels, img_labeled, lab_stats, _ = cv2.connectedComponentsWithStats(img_bin, connectivity=8, ltype=cv2.CV_32S)

    componentArray = [[] for _ in range(len(lab_stats)-1)]
    for component in range(1,len(lab_stats)):
        indices = np.where(img_labeled == component)
        indices = tuple(zip(*indices))
        indices = [t[::-1] for t in indices]
        componentArray[component-1].append(indices)

    return componentArray

def create_features(contours, DAG):

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
    def fits_poly(points, deg):
        is_vertical = False
        xs = [x for x,y in points]; ys = [y for x,y in points]
        if abs(points[0][0]-points[int(len(points)/2)][0])<50 and abs(points[0][0]-points[-1][0])<50: #Assumption
            print("isVertical")
            is_vertical = True
            tmp = xs, ys
            ys = tmp[0]; xs = tmp[1]
        xs_n = np.array(xs).reshape(len(xs), 1)
        ys_n = np.array(ys).reshape(len(ys), 1)
        min_rmse = 2

        x_train, x_test, y_train, y_test = train_test_split(xs_n, ys_n, test_size=0.3)

        # Train features
        poly_features = PolynomialFeatures(degree=deg, include_bias=False)
        x_poly_train = poly_features.fit_transform(x_train)


        # Linear regression
        poly_reg = LinearRegression()
        poly_reg.fit(x_poly_train, y_train)

        # Compare with test data
        x_poly_test = poly_features.fit_transform(x_test)
        poly_predict = poly_reg.predict(x_poly_test)
        poly_mse = mean_squared_error(y_test, poly_predict)
        poly_rmse = np.sqrt(poly_mse)

        print('Degree {} with RMSE {}'.format(deg, poly_rmse))

        polynomial = np.poly1d(np.polyfit(xs, ys, 2))
        ys2 = [polynomial(x) for x in xs]

        plt.plot(x_test, poly_predict, 'go')
        plt.plot(x_test, y_test, 'ro')
        plt.plot(xs, ys2, 'bo')
        ax = plt.gca()
        ax.set_ylim(ax.get_ylim()[::-1])
        plt.show()

        # Cross-validation of degree
        print(poly_rmse)
        if poly_rmse < min_rmse: #Need to check this assumption
            return True
        return False
    def smoothContours(contours):
        smoothened = []
        for contour in contours:
            x,y = contour.T
            # Convert from numpy arrays to normal arrays
            x = x.tolist()[0]
            y = y.tolist()[0]
            # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
            tck, u = splprep([x,y], u=None, s=1.0, per=1)
            # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
            u_new = numpy.linspace(u.min(), u.max(), 25)
            # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
            x_new, y_new = splev(u_new, tck, der=0)
            # Convert it back to numpy format for opencv to be able to display it
            res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new,y_new)]
            smoothened.append(numpy.asarray(res_array, dtype=numpy.int32))
        return smoothened
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
    def calc_quadratic_bezier_controlPoints(polynomial, x1, x2):

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
        return x_cp, y_cp

    final_featuresList = []

    #for c in range(len(contours)):
    for c in range(2,3):
        contour_points = contours[c][0]
        print(contour_points[0:10])
        feature = findAssociatedFeature(contour_points, DAG)
        if collinear_exact(contour_points):
            lineObj = Line(feature.x1, feature.y1, feature.x2, feature.y2, feature.orderNum)
            final_featuresList.append(lineObj)
        elif fits_poly(contour_points, 2):
            xs = [x for x,y in contour_points]; ys = [y for x,y in contour_points]
            if abs(contour_points[0][0]-contour_points[int(len(contour_points)/2)][0])<50 and abs(contour_points[0][0]-contour_points[-1][0])<50: #Assumption
                tmp = xs, ys
                ys = tmp[0]; xs = tmp[1]
            polynomial = np.poly1d(np.polyfit(xs, ys, 3))
            ys = [polynomial(x) for x in xs]
            x_cp, y_cp = calc_quadratic_bezier_controlPoints(polynomial,feature.x1,feature.x2)
            curveObj = Curve(feature.x1, feature.y1, x_cp, y_cp, feature.x2, feature.y2, feature.orderNum)
            final_featuresList.append(curveObj)
        else:
            print("Third degree polynomial curve")
    '''        xs = [x for x,y in contour_points]; ys = [y for x,y in contour_points]
            if abs(contour_points[0][0]-contour_points[int(len(contour_points)/2)][0])<50 and abs(contour_points[0][0]-contour_points[-1][0])<50: #Assumption
                tmp = xs, ys
                ys = tmp[0]; xs = tmp[1]
            polynomial = np.poly1d(np.polyfit(xs, ys, 4))
            ys = [polynomial(x) for x in xs]
            second_deriv = np.polyder(polynomial, 2)
            ys_prime = [second_deriv(x) for x in xs]

            def solve_for_y(poly_coeffs, y):
                pc = poly_coeffs.copy()
                pc[-1] -= y
                roots = np.roots(pc)
                roots = [r for r in roots if r>0]
                return roots

            x_vert = solve_for_y(np.polyfit(xs, ys_prime, 2), 0)  #find first derivative and then only use second derivative between those two points
            print(x_vert)
            y_vert = polynomial(x_vert)

            def divide_contour(ps, x):
                sps = sorted(ps, key=operator.itemgetter(0))
                firstComponent = [p for p in sps if p[0]<=x_vert]
                secondComponent = [p for p in sps if p[0]>x_vert]
                xs1 = [x for x,y in firstComponent]; ys1 = [y for x,y in firstComponent]
                xs2 = [x for x,y in secondComponent]; ys2 = [y for x,y in secondComponent]
                return xs1, ys1, xs2, ys2

            xs1, ys1, xs2, ys2 = divide_contour(contour_points, x_vert)
            print(xs1, ys1)

            polynomial_c1 = np.poly1d(np.polyfit(xs1, ys1, 3))
            polynomial_c2 = np.poly1d(np.polyfit(xs2, ys2, 3))
            ys1 = [polynomial_c1(x) for x in xs1]
            ys2 = [polynomial_c2(x) for x in xs2]
            x_cp1, y_cp1 = calc_quadratic_bezier_controlPoints(polynomial_c1,feature.x1, x_vert)
            x_cp2, y_cp2 = calc_quadratic_bezier_controlPoints(polynomial_c2, x_vert, feature.x2)
            curveObj = Curve(feature.x1, feature.y1, x_cp1, y_cp1, x_vert, y_vert, feature.orderNum)
            curveObj = Curve( x_vert, y_vert, x_cp2, y_cp2, feature.x2, feature.y2, feature.orderNum+.5)
            final_featuresList.append(curveObj1)
            final_featuresList.append(curveObj2)'''

    return final_featuresList

def createSubFeatures(featureList, vertices, image):



    def CalcBezierControlPoints(points):

        points = [(int((p1[0]+p2[0]+p3[0]+p4[0])/4),int((p1[1]+p2[1]+p3[1]+p4[1])/4)) for p1,p2,p3,p4 in zip(points[0::4], points[1::4], points[2::4], points[3::4])]

        firstPoint = points[0]
        lastPoint = points[-1]

        xs = [x for x,y in points]
        ys = [y for x,y in points]
        cubicPoly = np.poly1d(np.polyfit(xs, ys, 3))

        xDiff = lastPoint[0] - firstPoint[0]
        x1 = firstPoint[0] + xDiff / 3.0
        x2 = firstPoint[0] + 2.0 * xDiff / 3.0

        y1 = cubicPoly(x1)
        y2 = cubicPoly(x2)

        f1 = 0.296296296296296296296; # (1-1/3)^3
        f2 = 0.037037037037037037037; # (1-2/3)^3
        f3 = 0.296296296296296296296; # (2/3)^3

        b1 = y1 - firstPoint[1] * f1 - lastPoint[1] / 27.0;
        b2 = y2 - firstPoint[1] * f2 - f3 * lastPoint[1];

        c1 = (-2 * b1 + b2) / -0.666666666666666666;
        c2 = (b2 - 0.2222222222222 * c1) / 0.44444444444444444;

        secondPoint = (int(x1),int(c1))
        thirdPoint = (int(x2), int(c2))

        return firstPoint, secondPoint, thirdPoint, lastPoint


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
           lineMeasure = line_ext_map[str(lineObj.distance)],int(abs(lineObj.degrees))
           lineImage = line_Image_map[str(lineMeasure)]
           #print("line mapped to: ", lineImage)
           imageList.append(lineImage)
        else:
            #The curve end points aren't correct and then need to re-align them into proper DAG order
            #Then check that the Seamly2d VERSION RENDERS SOMEWHAT CLOSE TO THE REAL IMAGE

            #cps, polys = findControlPoints(c_tup)
            cps = CalcBezierControlPoints(c_tup)
            feature = findAssociatedFeature(c_tup)
            cv2.line(image, cps[0], cps[1] , (0,255,0), 5)
            cv2.line(image, cps[2], cps[3], (0,255,0), 5)
            #for p in polys: cv2.circle(image,p, 1, (0,255,0), -1)
            for p in c_tup: cv2.circle(image,p, 1, (0,0,255), -1)
            curveObj = Curve(cps[0][0], cps[0][1], cps[1][0], cps[1][1], cps[2][0],cps[2][1], cps[3][0], cps[3][1], feature.orderNum)
            #print("Curve: (a1,d1,a2,d2)", curveObj.degrees1, curveObj.distance1, curveObj.degrees2, curveObj.distance2)
            #print("Curve: (x1,y1,x2,y2)", curveObj.x1, curveObj.y1, curveObj.x2, curveObj.y2)
            feaList.append(curveObj)
            curveMeasure = curve_ext_map[str(curveObj.degrees1)+", "+str(curveObj.distance1)]
            curveImage = curve_image_map[str(curveMeasure)]
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

    DEMO = False

    argv = sys.argv[1:]
    arg_length = len(argv)
    if arg_length != 0:
        input_Image = argv[0]
        output_file = argv[1]
    else:
        print("Error: not enough argument supplied: \n compiler.py <path> <file name>")
        exit(0)

    line_Image_map, curve_image_map, line_ext_map, curve_ext_map = loadDictionaries()

    if DEMO:
        cv2.imshow('input',cv2.imread(input_Image))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    imageCleaned = clean_Image(input_Image)
    if DEMO:
        cv2.imshow('cleaned',imageCleaned)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    image_vertices = retrieveVertices(imageCleaned)
    if DEMO:
        fresh = imageCleaned.copy()
        for v in image_vertices:
            cv2.circle(fresh, (v[0], v[1]), 10, (0,255,0), -1)
        cv2.imshow('vertices',fresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    DAG = createDAG(image_vertices)
    if DEMO:
        for f in DAG:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(fresh,str(f.orderNum),(int((f.x1+f.x2)/2),int((f.y1+f.y2)/2)), font, 1,(0,255,0),2,cv2.LINE_AA)
        cv2.imshow('DAG',fresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    contours = extract_initial_contours(image_vertices, imageCleaned) #Order of contours doesn't matter here
    if DEMO:
        fresh[fresh == 0] = 255
        colors = [(255, 255, 0),(128, 128, 0),(192, 57, 43), (255,0,0), (35, 155, 86)]
        for c in range(len(contours)):
            indices = contours[c][0]
            print(indices[0])
            for i in indices:
                cv2.circle(fresh, (i[0], i[1]), 1, colors[c], -1)
        cv2.imshow('Contours',fresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #set_points = []
    #for p in contours[2][0]:
    #    add = True
    #    for sp in set_points:
    #        if p[0] == sp[0]:
    #            add = False
    #    if add == True:
    #        set_points.append(p)
    #print(set_points)
    #xs = [x for x,y in set_points]; ys = [700-y for x,y in set_points]
    #contours[2][0] = set_points
    #plt.plot(xs, ys, 'bo')
    #plt.show()
    features = create_features(contours, DAG)


    #print(features)


    #subFeatureList, imageList, height = createSubFeatures(featuresList, image_vertices, imageCleaned)
    #if DEBUG_FLAG:
    #    for f in subFeatureList:
    #        print("feature: (x1,y1,x2,y2)", f.endpoints)
    #compiledXML = compileXML(subFeatureList, output_file, height)
    #buildImageRepresentation(imageList) #CANT DO THIS UNTIL THE DICTIONARY IS BUILT OUT
