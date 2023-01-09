# https://stackoverflow.com/questions/74416690/access-values-in-dict-in-tf-dataset-dataset-map-with-tf-striing

import os
from os.path import exists

import tensorflow as tf
import tensorflow_io as tfio
from google.protobuf import json_format

import matplotlib.pyplot as plt

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
None
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(physical_devices)
print(config)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Variables
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""			
image = tf.io.read_file( "F:\\Pictures\\Kids\\images.jpg" )
image = tf.io.decode_jpeg(image)							
image = tf.cast( image, dtype=tf.int64 )
					
example = tf.train.Example(
	features=tf.train.Features(
		feature={
			"1": tf.train.Feature(
			
				int64_list=tf.train.Int64List(
					value=tf.constant( image, shape=( 183 * 275 * 3 ) ).numpy() )
					
					),
			"2": tf.train.Feature(
			
				int64_list=tf.train.Int64List(
					value=tf.constant( image, shape=( 183 * 275 * 3 ) ).numpy() )
					
					)		
			}))

data_string = json_format.MessageToJson(example)
example_binary = tf.io.decode_json_example(data_string)

example_phase = tf.io.parse_example(
	serialized=[example_binary.numpy()],
	features = { 	"1": tf.io.FixedLenFeature(shape=[ 183 * 275 * 3 ], dtype=tf.int64),
					"2": tf.io.FixedLenFeature(shape=[ 183 * 275 * 3 ], dtype=tf.int64)
				})

data = list(example_phase.items())
label = [ int(x[0]) for x in data ]

data = [ x[1] for x in data ]
data = [ x[0] for x in data ]

dataset = tf.data.Dataset.from_tensors(( data, label ))



plt.imshow( tf.constant( example_phase['1'], shape=( 183, 275, 3) ) )
plt.show()
