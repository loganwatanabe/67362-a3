
#to read entire label files
def read_from_file(filename):
    image_vals = []
     
    image_vals = [int(line.strip()) for line in open(filename)]

    return image_vals

#get image at n
img_height = 28
img_width = 28

def show_nth_image(fname, n):
    f = open(fname, "rU")
    img = []
    # skip first n-1 images
#    f.seek( (n-1)*img_height )
    for i in range( (n-1)*img_height):
        f.readline()
    for i in range(img_height):
        line = f.readline().rstrip('\n')
        img.append(line)
    return img

#get label at n
def show_nth_label(fname, n):
    f = open(fname, "rU")
    # skip first n-1 images
#    f.seek( (n-1)*img_height )
    for i in range(n-1):
        f.readline()
    label = int(f.readline().rstrip('\n'))
    return label

def getPixels_nth(filename, n):
  digit = show_nth_image(filename, n)
  pixel_array = []

  for i in range(0, 28):
    row_array = list(digit[i])
    pixel_array.append(row_array)
  return pixel_array

def total_count(filename, n):
    training_labels = read_from_file(filename)
    total_counts = []
    for i in range(0, 10):
        total_counts.append(training_labels.count(i))
    return total_counts[n]


import re
import math

def getpixels(image):
  digit = image
  pixel_array = []

  for i in range(0, 28):
    row_array = list(digit[i])
    # pixel_array.append(row_array)
    pixel_array = pixel_array + row_array
  return dict([(i,1) for x in pixel_array])

  # def getwords(doc):
  # splitter=re.compile('\\W*')
  # # print( doc )
  # # Split the words by non-alpha characters
  # words=[s.lower() for s in splitter.split(doc) 
  #         if len(s)>2 and len(s)<20]
  
  # # Return the unique set of words only
  # return dict([(w,1) for w in words])


class classifier:
  def __init__(self,getfeatures,filename=None):
    # feature is an INT that represents the index of the pixel in the 784-long array
    # Counts of feature/category combinations
    self.fc={}
    # Counts of images in each category
    self.cc={}
    self.getfeatures=getfeatures
    
  # Increase the count of a feature/category pair
  def incf(self,f,cat):
    self.fc.setdefault(f,{})
    self.fc[f].setdefault(cat,0)
    self.fc[f][cat]+=1

  # Increase the count of a category
  def incc(self,cat):
    self.cc.setdefault(cat,0)
    self.cc[cat]+=1
 
  # The number of times a feature has appeared in a category
  def fcount(self,f,cat):
    if f in self.fc and cat in self.fc[f]: 
      return float(self.fc[f][cat])
    return 0.0
  
  # The number of items in a category
  def catcount(self,cat):
    if cat in self.cc:
      return float(self.cc[cat])
    return 0
  
  # The total number of items
  def totalcount(self):
    return sum(self.cc.values())

  # The list of all categories
  def categories(self):
    return self.cc.keys()

  def train(self,item,cat):
    features=self.getfeatures(item)
    # Increment the count for every feature with this category
    for f in features:
      self.incf(f,cat)

    # Increment the count for this category
    self.incc(cat)
  
  def fprob(self,f,cat):
    if self.catcount(cat)==0: return 0

    # The total number of times this feature appeared in this 
    # category divided by the total number of items in this category
    return self.fcount(f,cat)/self.catcount(cat)

  def weightedprob(self,f,cat,prf,weight=1.0,ap=0.5):
    # Calculate current probability
    basicprob=prf(f,cat)

    # Count the number of times this feature has appeared in
    # all categories
    totals=sum([self.fcount(f,c) for c in self.categories()])

    # Calculate the weighted average
    bp=((weight*ap)+(totals*basicprob))/(weight+totals)
    return bp




class naivebayes(classifier):
  
  def __init__(self,getfeatures):
    classifier.__init__(self,getfeatures)
    self.thresholds={}
  
  def docprob(self,item,cat):
    features=self.getfeatures(item)   

    # Multiply the probabilities of all the features together
    p=1
    for f in features: p*=self.weightedprob(f,cat,self.fprob)
    return p

  def prob(self,item,cat):
    catprob=self.catcount(cat)/self.totalcount()
    docprob=self.docprob(item,cat)
    return docprob*catprob
  
  def setthreshold(self,cat,t):
    self.thresholds[cat]=t
    
  def getthreshold(self,cat):
    if cat not in self.thresholds: return 1.0
    return self.thresholds[cat]
  
  def classify(self,item,default=None):
    probs={}
    # Find the category with the highest probability
    max=0.0
    for cat in self.categories():
      probs[cat]=self.prob(item,cat)
      if probs[cat]>max: 
        max=probs[cat]
        best=cat

    # Make sure the probability exceeds threshold*next best
    for cat in probs:
      if cat==best: continue
      if probs[cat]*self.getthreshold(best)>probs[best]: return default
    return best