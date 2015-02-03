import re
import math

def getpixels(image):
    digit = image
    pixel_array = []
    col = {}
    
    for i in range(0, 28):
        row_array = list(digit[i])
        pixel_array = pixel_array + row_array #784 long array

    for index in range(0,len(pixel_array)):
        col[index]= {}
        col[index]["value"] = pixel_array[index]
    return col


class classifier:
  def __init__(self,getfeatures,filename=None):
    # feature is an INT that represents the index of the pixel in the 784-long array
    # Counts of feature/category combinations
    self.fc={}
    # Counts of images in each category
    self.cc={}
    self.getfeatures=getfeatures
    
  # Increase the count of a feature-value/category pair
  def incf(self,f,val,num):
    self.fc.setdefault(f,{})
    self.fc[f].setdefault(num,{})
    self.fc[f][num].setdefault(val,0)
    self.fc[f][num][val]+=1

  # Increase the count of a category
  def incc(self,num):
    self.cc.setdefault(num,0)
    self.cc[num]+=1
 
  # The number of times a feature-value pair has appeared in a category
  def fcount(self,f,val,num):
    if f in self.fc and num in self.fc[f] and val in self.fc[f][num]: 
      return float(self.fc[f][num][val])
    return 0.0
  
  # The number of items in a category
  def catcount(self,num):
    if num in self.cc:
      return float(self.cc[num])
    return 0
  
  # The total number of items
  def totalcount(self):
    return sum(self.cc.values())

  # The list of all categories
  def categories(self):
    return self.cc.keys()

  def train(self,item,num):
    features=self.getfeatures(item)
    # Increment the count for every feature with this category
    for f in features:
      self.incf(f,features[f]["value"],num)

    # Increment the count for this category
    self.incc(num)
    
  def cprob(self,cat):
    if self.catcount(cat)==0: return 0

    # The total number of times this feature appeared in this 
    # category divided by the total number of items in this category
    return self.catcount(cat)/self.totalcount()

  def fprob(self,f,val,cat):
    if self.catcount(cat)==0: return 0

    # The total number of times this feature appeared in this 
    # category divided by the total number of items in this category
    return self.fcount(f,val,cat)/self.catcount(cat)

  def weightedprob(self,f,val,cat,k=1.0,v=3):
    # def weightedprob(self,f,val,cat,prf,k=1.0,v=3):
    # Calculate current probability
    # basicprob=prf(f,val,cat)

    # Count the number of times this feature has appeared in
    # all categories
    # totals=sum([self.fcount(f,val,c) for c in self.categories()])

    # Calculate the weighted average
    # bp=((weight*ap)+(totals*basicprob))/(weight+totals)
    bp = (self.fcount(f,val,cat)+k)/(self.catcount(cat)+(k*v))
    return bp




class naivebayes(classifier):
  
  def __init__(self,getfeatures):
    classifier.__init__(self,getfeatures)
    self.thresholds={}
  
  def docprob(self,item,cat): #this is returning negative numbers cuz log(decimal)
    features=self.getfeatures(item)   

    # Multiply the probabilities of all the features together
    p = math.log(self.cprob(cat))
    for f in features:
        p += math.log(self.weightedprob(f,features[f]["value"],cat))
        # p += math.log(self.weightedprob(f,features[f]["value"],cat,self.fprob))
        
        # HELP HERE, IS THIS HOW WE GET PROB?
    return p*(-1)/100

  def prob(self,item,cat):
    catprob=self.catcount(cat)/self.totalcount()
    docprob=self.docprob(item,cat)
    return 1-(docprob*catprob)
    # return docprob*catprob
  
  def setthreshold(self,cat,t):
    self.thresholds[cat]=t
    
  def getthreshold(self,cat):
    if cat not in self.thresholds: return 1.0
    return self.thresholds[cat]
  
  def classify(self,item,default=None):
    probs={}
    # Find the category with the highest probability
    max=0.0
    best = default
    for cat in self.categories():
      probs[cat]=self.prob(item,cat)
      # print"p(",cat, ") = ", probs[cat]
      if probs[cat]>max: 
        max=probs[cat]
        best=cat

    # Make sure the probability exceeds threshold*next best
    for cat in probs:
      if cat==best: continue
      if probs[cat]*self.getthreshold(best)>probs[best]: return default
    return best