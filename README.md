# json_data_set
For study support communication messages transferring ( that is becase the sample is hard and they keep asking about this methods when Google JSON is corect but not alway working )

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
