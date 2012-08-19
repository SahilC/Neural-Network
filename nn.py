import math,random,string
#sigmoidal function
def sig(x):
  return math.tanh(x)

#Derivative of Sigmoidal funtion
def dsig(y):
  return 1 - y**2

#Build Matrix
def makeMatrix ( I, J, fill=0.0):
  m = []
  for i in xrange(I):
    m.append([fill]*J)
  return m
  
def randomizeMatrix ( matrix, a, b):
  for i in xrange ( len (matrix) ):
    for j in xrange ( len (matrix[0]) ):
      matrix[i][j] = random.uniform(a,b)

class NeuralNet:
  #initialize the neural net
  def __init__(self, NIp, NHid, NOut):

    self.ni = NIp + 1 
    self.nh = NHid
    self.no = NOut
    self.ai, self.ah, self.ao = [],[], []
    self.ai = [1.0]*self.ni
    self.ah = [1.0]*self.nh
    self.ao = [1.0]*self.no
    self.wi = makeMatrix (self.ni, self.nh)
    self.wo = makeMatrix (self.nh, self.no)
    randomizeMatrix ( self.wi, -0.2, 0.2 )
    randomizeMatrix ( self.wo, -2.0, 2.0 )
    self.ci = makeMatrix (self.ni, self.nh)
    self.co = makeMatrix (self.nh, self.no)
    
  def run(self, inputs):
    if len(inputs) != self.ni-1:
      print 'incorrect number of inputs'
    
    for i in xrange(self.ni-1):
      self.ai[i] = inputs[i]
      
    for j in xrange(self.nh):
      sum = 0.0
      for i in xrange(self.ni):
        sum +=( self.ai[i] * self.wi[i][j] )
      self.ah[j] = sig(sum)
    
    for k in xrange(self.no):
      sum = 0.0
      for j in xrange(self.nh):        
        sum +=( self.ah[j] * self.wo[j][k] )
      self.ao[k] = sig(sum)
      
    return self.ao
      
      
  #backprop for network
  def backPropagate (self, targets, N, M):
    output_deltas = [0.0] * self.no
    for k in xrange(self.no):
      error = targets[k] - self.ao[k]
      output_deltas[k] =  error * dsig(self.ao[k]) 
    for j in xrange(self.nh):
      for k in xrange(self.no):
        change = output_deltas[k] * self.ah[j]
        self.wo[j][k] += N*change + M*self.co[j][k]
        self.co[j][k] = change
    hidden_deltas = [0.0] * self.nh
    for j in xrange(self.nh):
      error = 0.0
      for k in xrange(self.no):
        error += output_deltas[k] * self.wo[j][k]
      hidden_deltas[j] = error * dsig(self.ah[j])
    for i in xrange (self.ni):
      for j in xrange (self.nh):
        change = hidden_deltas[j] * self.ai[i]
        self.wi[i][j] += N*change + M*self.ci[i][j]
        self.ci[i][j] = change
    error = 0.0
    for k in xrange(len(targets)):
      error = 0.5 * (targets[k]-self.ao[k])**2
    return error
        
  #display neural weights of the neuron      
  def weights(self):
    print 'Input weights:'
    for i in xrange(self.ni):
      print self.wi[i]
    print
    print 'Output weights:'
    for j in xrange(self.nh):
      print self.wo[j]
    print ''
  #Run A pattern through the Neural Net
  def test(self, patterns):
    print 'Inputs\t\tValue\t\t\tTarget'
    for p in patterns:
      inputs = p[0]
      print  p[0], ':', self.run(inputs), '\t', p[1]
  #Train Network w.r.t. the provided input and target values
  def train (self, patterns, max_iterations = 1000, N=0.5, M=0.1):
    for i in xrange(max_iterations):
      for p in patterns:
        inputs = p[0]
        targets = p[1]
        self.run(inputs)
        error = self.backPropagate(targets, N, M)
      if i % 50 == 0:
        print 'Error:', error
    self.test(patterns)
#Build Neural Net and Run Some tests
def tests():
  pat = [
      [[0,0,0,0], [1]],
      [[0,0,0,1],[0]],
      [[0,1,0,0],[0]],
      [[0,0,1,0],[0]],
      [[0,0,1,1],[1]],
      [[0,1,0,1],[1]],
      [[1,0,0,1],[1]],
      [[1,0,1,0],[1]],
      [[1,1,0,0],[1]],
      [[0,1,1,0],[1]],
      [[1,1,1,0],[0]],
      [[1,0,1,1],[0]],
      [[0,1,1,1],[0]],
      [[1,1,0,1],[0]],
  ]
  nn = NeuralNet( 4, 5, 1)
  nn.train(pat)
  nn.weights()
  print 'Test'
  testr=[[[1,0,1,0],[1]],[[1,1,0,1],[0]],[[0,1,1,1],[0]]]
  nn.test(testr)
  
if __name__ == "__main__":
    tests()
