import sys
import tensorflow as tf
import numpy as np
import model_cache as cache

def predict(model_name, image_path):
    # Load the image, models and the lables
    graph_def = load_graph(model_name)
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    label_lines = load_labels(model_name)

    reset_graph()

    # Disable GPU
    config = tf.ConfigProto(
            device_count = {'GPU': 0}
    )
    with tf.Session(config=config) as sess:
        # Load and import the graph within the session 
        _ = tf.import_graph_def(graph_def, name='')

        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        
        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})
        
        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        
        classify_result = []
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            result = {
                'label':human_string,
                'score':np.asscalar(score)
            }
            classify_result.append(result)
            print result
        
        return {
            'model':model_name,
            'labels':classify_result
            }

def reset_graph():
    tf.reset_default_graph()

def load_graph(model_id):
    cached_model = cache.get(model_id)
    if cached_model == None:
        # Unpersists graph from file
        with tf.gfile.FastGFile("inference/" + model_id + "/graph.pb", 'rb') as f:
            print 'Loading graph from file'
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            cache.put(model_id, graph_def)
            return graph_def
    else:
        print 'Loading graph from cache'
        return cached_model;

def load_labels(model_id):
  return [line.rstrip() for line in tf.gfile.GFile("inference/" + model_id + "/labels.txt")]

if __name__ == '__main__':
    print("model:" + sys.argv[1])
    print("image:" + sys.argv[2])
    classify(sys.argv[1], sys.argv[2])
