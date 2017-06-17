from werkzeug.contrib.cache import SimpleCache

cache = SimpleCache()

def get(model_id):
  return cache.get(model_id)

def put(model_id, model):
  return cache.set(model_id, model)