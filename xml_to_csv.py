"""
xml_to_csv
generate xml file, which is generated diriectly from labelImag,
to csv, ready for csv-to-tfRecord transformation.
Can be called by ./generate_tfrecord.sh
from cat_dataset directory
"""
import os
import glob # an interesting tool for file operations
import pandas as pd
import xml.etree.ElementTree as ET
import sys, getopt # for processing command line parameters

'''
print_usage
'''
def print_usage():
    print("Generate csv record from xml files.")
    print("-h or --help:    usage")
    print("--trainxml  :    the xml file path for train records")
    print("--testxml   :    the xml file path for test records")
    print("--traincsv  :    the generated csv record path for train records")
    print("--testcsv   :    the generated csv record path for test records")
    print("All of the 4 paremeters above should be specified.")

''' 
xml_to_csv
Input: path: the path of xml files
'''
def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

'''
main
'''
def main():
    # get command line parameters
    try:
        options, args = getopt.getopt(sys.argv[1:], "h", ["help", "trainxml=", "testxml=", "traincsv=", "testcsv="])
    except getopt.GetoptError:
        sys.exit()
    # parse the cmd line parameters
    for name, value in options:
        if name == "--help":
            print_usage()
        if name == "--trainxml":
            TRAIN_XML_DIR = value
        if name == "--testxml":
            TEST_XML_DIR = value 
        if name == "--traincsv":
            TRAIN_CSV_DIR = value
        if name == "--testcsv":
            TEST_CSV_DIR = value
    if not TRAIN_XML_DIR or not TEST_XML_DIR or not TRAIN_CSV_DIR or not TEST_CSV_DIR:
        print("wrong input parameters!")
        print_usage()

    # convert train images
    # the path of xml files
    image_path = os.path.join(os.getcwd(), TRAIN_XML_DIR)
    xml_df = xml_to_csv(image_path)
    # write to the path of csv files
    xml_df.to_csv(TRAIN_CSV_DIR, index=None)
    print('Successfully converted train xml to csv.')

    # convert test images 
    # the path of xml files
    image_path = os.path.join(os.getcwd(), TEST_XML_DIR)
    xml_df = xml_to_csv(image_path)
    # write to the path of csv files
    xml_df.to_csv(TEST_CSV_DIR, index=None)
    print('Successfully converted test xml to csv.')

if __name__ == '__main__':
    main()
