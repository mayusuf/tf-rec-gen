#===========================================================================================
#
#title           :data_parse.py
#description     :This script read json file from a directory.
#author		 	 :Abu Yusuf and FusionSystems
#date            :20190809
#version         :1.0    
#usage		     :python data_parse.py
#notes           :Install minimum python 3.6, Python Image Library.
#
#============================================================================================

import sys
import os
import json
import io

import tensorflow as tf
from object_detection.utils import dataset_util


from PIL import Image  # Python image library

os.chdir(os.getcwd())

lableFolder="DaimlerValDatasetLabel"
imageFolder="DaimlerValDatasetImage"
#recordFolder="tfRecord/train.record"
valRecordFolder="tfRecord/val.record"

flags = tf.app.flags
#flags.DEFINE_string('output_path', recordFolder, 'Path to output TFRecord')
flags.DEFINE_string('output_valpath', valRecordFolder, 'Path to output TFRecord')
FLAGS = flags.FLAGS



def create_tf_example(example):
    # TODO(user): Populate the following variables from your example.
    height = example["height"] # Image height
    width = example["width"] # Image width
    filename = example["filename"].encode('utf8') # Filename of the image. Empty if image is not from file
    
    img_path = imageFolder+os.sep+example["filename"]
    # imgByteArr = io.BytesIO()
    # encoded_image_data =  imgByteArr.getvalue()  # Encoded image bytes
    
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_image_data = fid.read()
        
    image_format =  'png' # b'jpeg' or b'png'
    
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    
    for x_y_coordinates in example["bbox"]:
        if(x_y_coordinates["identity"] == "cyclist"):
            xmins.append(float(x_y_coordinates["mincol"]) / width) # List of normalized left x coordinates in bounding box (1 per box)
            xmaxs.append(float(x_y_coordinates["maxcol"]) / width) # List of normalized right x coordinates in bounding box(1 per box)
            ymins.append(float(x_y_coordinates["minrow"]) / height) # List of normalized top y coordinates in bounding box (1 per box)
            ymaxs.append(float(x_y_coordinates["maxrow"]) / height) # List of normalized bottom y coordinates in bounding box (1 per box)
            
            classes_text.append(x_y_coordinates["identity"].encode('utf8')) # List of string class name of bounding box (1 per box)
            classes.append(1) # List of integer class id of bounding box (1 per box)
        
        # elif(x_y_coordinates["identity"] == "motorcyclist"):
            # classes.append(2) # List of integer class id of bounding box (1 per box)
        # elif(x_y_coordinates["identity"] == "tricyclist"):
            # classes.append(3) # List of integer class id of bounding box (1 per box)
        # elif(x_y_coordinates["identity"] == "pedestrian"):
            # classes.append(4) # List of integer class id of bounding box (1 per box)
        # elif(x_y_coordinates["identity"] == "wheelchairuser"):
            # classes.append(5) # List of integer class id of bounding box (1 per box)
        # elif(x_y_coordinates["identity"] == "mopedrider"):
            # classes.append(6) # List of integer class id of bounding box (1 per box)

    print(xmins)
    print(xmaxs)
    print(ymins)
    print(ymaxs)
    print(classes_text)
    print(classes)
    
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example



      
def main(_):
    
    #print(FLAGS.output_path)
    writer = tf.python_io.TFRecordWriter(FLAGS.output_valpath)
    
    #writer1 = tf.python_io.TFRecordWriter(FLAGS.output_valpath)
    
    # TODO(user): Write code to read in your dataset to examples variable
    
    jsonFiles=os.listdir(lableFolder);

    #print(jsonFiles)

    #number=0
    
    #hasSecLevel = 0
    
    # Read directory contains json files one by one
    for file in jsonFiles:
        
        #if number == 20:
            #break    # break here
        #print(file)
        
        # Read string of a json file
        with open(lableFolder+os.sep+file) as jsonString: 
            
            json_data = json.load(jsonString)
            #print(json_data['imagename'])
            
            imInfo = Image.open(imageFolder+os.sep+json_data["imagename"])
            w, h = imInfo.size
            
            # print('width: ', w)
            # print('height:', h)
            # print(json_data["children"][0]["mincol"])
            # print(json_data["children"][0]["minrow"])
            # print(json_data["children"][0]["maxcol"])
            # print(json_data["children"][0]["maxrow"])
                        
            example = {
                "filename": json_data['imagename'],
                "width":w,
                "height":h,
                "bbox":json_data["children"]
            }
            
            #print(example["bbox"])
            
            # checking is there any second level children/ bounding boxes
            # for bb_info in example["bbox"]:
                # if(len(bb_info["children"])>0):
                    # hasSecLevel +=1
                   
            print("==================") 
            #if(number<150):
            
            tf_example = create_tf_example(example)
            writer.write(tf_example.SerializeToString())
            
            # else:
                # tf_example = create_tf_example(example)
                # writer1.write(tf_example.SerializeToString())
                
            #number = number + 1
            
            #for example in examples:
            # tf_example = create_tf_example(example)
            # writer.write(tf_example.SerializeToString())
            
    #print("\n Is there any second level children:",hasSecLevel)
    
    writer.close()
    
    #writer1.close()


if __name__ == '__main__':
  tf.app.run()