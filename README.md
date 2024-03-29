# json_data_set
For study support communication messages transferring ( that is because the sample is hard to understand and they keep asking about this method when Google JSON is correct but not always working )

- Saved to text file required you to convert to string format, file write is only print.
- Alternate using .json format.

## Sample StateLess messages ##

Sample of StateLess meesages, transfrom messages communication with sequence and structures since UDP or Wireless signals is not garantee order one way is AgentQueue at the remote device, communication devices and the communication data.

```
🔍 No wall free for all { ⬆️, ➡️, ⬅️, ⬇️ }
🔍 Left wall { ⬆️, ➡️, ⬇️ }, { ⬆️, ⬇️ }, { ⬇️ }, { ⬆️ }
🔍 Right wall { ⬆️, ⬅️, ⬇️ }, { ⬆️, ⬇️ }, { ⬇️ }, { ⬆️ }
🔍 Top wall { ➡️, ⬅️, ⬇️ }, { ➡️, ⬅️ }, { ⬅️ }, { ➡️ }
🔍 Buttom wall { ⬆️, ➡️, ⬅️ }, { ➡️, ⬅️ }, { ⬅️ }, { ➡️ }
```

## Build dataset in .JSON format ##

Google .json required attribute mapping.

```
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
```

## Decode dataset in .JSON format ##

From communiations as .JSON to dataset.

```
data_string = json_format.MessageToJson(example)
example_binary = tf.io.decode_json_example(data_string)

example_phase = tf.io.parse_example(
serialized=[example_binary.numpy()],
features = { 	
                "1": tf.io.FixedLenFeature(shape=[ 183 * 275 * 3 ], dtype=tf.int64),
                "2": tf.io.FixedLenFeature(shape=[ 183 * 275 * 3 ], dtype=tf.int64)
            })
```
## Usage ##

Data usage and extactions.

```
data = list(example_phase.items())
label = [ int(x[0]) for x in data ]

data = [ x[1] for x in data ]
data = [ x[0] for x in data ]

dataset = tf.data.Dataset.from_tensors(( data, label ))

plt.imshow( tf.constant( example_phase['1'], shape=( 183, 275, 3) ) )
plt.show()
```

## Result ##

Sample of data in text string format for large data input than print sting.


![Alt text](https://github.com/jkaewprateep/json_data_set/blob/main/06.png "Title")
