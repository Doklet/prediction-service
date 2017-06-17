class ModelDetails(object):

  def __init__(self, id, userid, name, provider, path):
    self.id = id
    self.userid = userid
    self.name = name
    self.provider = provider
    self.path = path

  def validate(self):
  	if self.path == None or self.path == 'undefined':
  		raise Exception('The path of the model is invalid: ' + self.path)