from werkzeug.contrib.cache import SimpleCache

cache = SimpleCache()

registry = dict()

def get(model_id):
	return registry.get(model_id, None)
  # return cache.get(model_id)

def put(model_id, model):
	registry[model_id] = model
  # return cache.set(model_id, model)