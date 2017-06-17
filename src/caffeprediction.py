import os
import caffe
import numpy as np
import pandas as pd
import exifutil
import labels

def predict(modeldetails, img_path):
  image = exifutil.open_oriented_im(img_path)
  labels = load_labels(modeldetails.path)
  classifier = load_classifier(modeldetails.path)
  scores = classifier.predict([image], oversample=True).flatten()
  print img_path
  classify_result = []
  for (label, score) in zip(labels, scores):
    result = {
      'label':label,
      'score':np.asscalar(score)
    }
    classify_result.append(result)
  return {
    'model':modeldetails,
    'labels':classify_result
    }


def load_classifier(path):
  model_def_file = path + "/deploy.prototxt"
  pretrained_model_file = path + "/weights.caffemodel"
  mean = load_mean(path)
  image_dim = 256
  raw_scale = 255
  return caffe.Classifier(
            str(model_def_file), str(pretrained_model_file),
            image_dims=(image_dim, image_dim), raw_scale=raw_scale,
            mean=mean.mean(1).mean(1), channel_swap=(2, 1, 0)
        )


def load_mean(path):
  blob = caffe.proto.caffe_pb2.BlobProto()
  data = open( path + '/mean.binaryproto' , 'rb' ).read()
  blob.ParseFromString(data)
  return np.array( caffe.io.blobproto_to_array(blob) )[0]

def load_labels(name):
  labels.load(name)

if __name__ == "__main__":
  img_path1 = './data/apple/bad/thumb_IMG_0573_1024.jpg'
  img_path2 = './data/apple/good/thumb_IMG_0610_1024.jpg'
  name = 'apple'
  result = classify(name, img_path2)
  print result