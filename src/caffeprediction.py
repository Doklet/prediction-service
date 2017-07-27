import os
import caffe
import numpy as np
import pandas as pd
import exifutil
import model_cache as cache
import labels
from timeit import default_timer as timer

caffe.set_device(0)
caffe.set_mode_gpu()

def predict(modeldetails, img_path):
  print('caffe prediction')
  print(modeldetails)
  image = exifutil.open_oriented_im(img_path)
  labellist = load_labels(modeldetails.path)
  labellist.reverse()
  # classifier = load_classifier(modeldetails)
  classifier = get_or_load_classifier(modeldetails)
  start = timer()
  scores = classifier.predict([image], oversample=True).flatten()
  end = timer()
  elapsed = (end - start)
  print('Caffe prediction time: ' + str(elapsed))
  print img_path
  classify_result = []
  print(labellist)
  print(scores)
  for (label, score) in zip(labellist, scores):
    result = {
      'label':label,
      'score':np.asscalar(score)
    }
    classify_result.append(result)
  return {
    'model':modeldetails.id,
    'labels':classify_result
    }


def get_or_load_classifier(modeldetails):
  cached_model = cache.get(modeldetails.id)
  if cached_model == None:
    deploy = modeldetails.path + "/deploy.prototxt"
    weights = modeldetails.path + "/weights.caffemodel"
    mean = load_mean(modeldetails.path)
    image_dim = 256
    raw_scale = 255
    classifier = caffe.Classifier(
              str(deploy), str(weights),
              image_dims=(image_dim, image_dim), raw_scale=raw_scale,
              mean=mean.mean(1).mean(1), channel_swap=(2, 1, 0)
          )
    cache.put(modeldetails.id, classifier)
    return classifier
  else:
    return cached_model

def load_classifier(modeldetails):
  deploy = modeldetails.path + "/deploy.prototxt"
  weights = modeldetails.path + "/weights.caffemodel"
  mean = load_mean(modeldetails.path)
  image_dim = 256
  raw_scale = 255
  classifier = caffe.Classifier(
            str(deploy), str(weights),
            image_dims=(image_dim, image_dim), raw_scale=raw_scale,
            mean=mean.mean(1).mean(1), channel_swap=(2, 1, 0)
        )
  return classifier

def load_mean(path):
  blob = caffe.proto.caffe_pb2.BlobProto()
  data = open( path + '/mean.binaryproto' , 'rb' ).read()
  blob.ParseFromString(data)
  return np.array( caffe.io.blobproto_to_array(blob) )[0]

def load_labels(path):
  return labels.load(path + '/labels.txt')

if __name__ == "__main__":
  img_path1 = './data/apple/bad/thumb_IMG_0573_1024.jpg'
  img_path2 = './data/apple/good/thumb_IMG_0607_1024.jpg'
  name = 'apple'
  result = classify(name, img_path2)
  print result