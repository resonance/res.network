'''
Converst RSL files into Seamly2D (XML Based) files to be opened and rendered in Seamly2D.
'''

from os import listdir

repeatLines = ["<?xml version=\"1.0\" encoding=\"UTF-8\"?>", "<pattern>", "<!--Pattern created with Seamly2D v0.6.0.0a (https://fashionfreedom.eu/).-->", "<version>0.6.0</version>",
                "<unit>inch</unit>", "<description/>","<notes/>","<measurements>/Users/theodoreseem/ResonanceHub/image2XML/image2XML/Data/Vector_Measurements/Tucker-Brianna.vit</measurements>",
                "<increments/>", "<draw name=\"Test1\">", "<calculation>", "<point id=\"1\" mx=\".1\" my=\".1\" name=\"A\" type=\"single\" x=\"90\" y=\"90\"/>",
                "<point angle=\"90\" basePoint=\"1\" id=\"2\" length=\"height*2\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"A1\" type=\"endLine\" typeLine=\"hair\"/>",
                "<point angle=\"180\" basePoint=\"1\" id=\"3\" length=\"height*2\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"A2\" type=\"endLine\" typeLine=\"hair\"/>",
                "<point id=\"4\" mx=\".1\" my=\".1\" name=\"B\" type=\"single\" x=\"20\" y=\"20\"/>"]
repeatLines2 = ["</calculation>","<modeling/>","<details/>","</draw>","</pattern>"]

master_Directory = '/Users/theodoreseem/res.Network/Curve_Integration/'

def compile(file):

    inputRSL = master_Directory + imageStore + "/" + file + ".gui"
    outputXML =  master_Directory + "xml_outputs/" + file + ".val"

    input = open(inputRSL, "r")
    output = open(outputXML, "w")

    startPoint = 4
    currentID = startPoint

    for line in repeatLines:
        output.write(line + "\n")

    for counter, line in enumerate(input, start=-4):
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
            output.write("<point angle=\"%d\" basePoint=\"%d\" id=\"%d\" length=\"%.1f\" lineColor=\"black\" mx=\".1\" my=\".1\" name=\"B1\" type=\"endLine\" typeLine=\"hair\"/>\n" %(angle, counter+startPoint, counter+startPoint+1, length/50))
            currentID = currentID + 1
    output.write("<line firstPoint=\"%d\" id=\"%d\" lineColor=\"black\" secondPoint=\"%d\" typeLine=\"hair\"/>\n" %(currentID, currentID+1, startPoint))


    for line in repeatLines2:
        output.write(line + "\n")

    output.close()

if __name__ == "__main__":

    argv = sys.argv[1:]
    arg_length = len(argv)
    if arg_length != 0:
        if argv[0] == "single":
            imgStore = argv[1]
        else:
            imgStore = argv[1]
            files = [f[:-4] for f in listdir(master_Directory + imgStore) if f != '.DS_Store']
    else:
        print("Error: not enough argument supplied: \n Processing.py <single/multi> <Image/Image_Directory>")
        exit(0)

    if argv[0] == "single":
        file = imgStore[:-4]
        print("Compiling: ", file)
        compile(file, "")
    else:
        for count, file in enumerate(files):
            print("#", count, "- Compiling: ", file)
            compile(file, imgStore)
