cl="cl"

from digitclass import *
from helpers import *
def test_env(images, labels):
    global cl
    cl=classifier(getpixels)
    for i in range(1,2):
        cl.train(show_nth_image(images, i), show_nth_label(labels, i))
    print cl.fc
    
def train(cl, images, labels):
    # global cl
    # cl=classifier(getpixels)
    label_length = len([int(line.strip()) for line in open(labels)])
    for i in range(1, 1001):
        cl.train(show_nth_image(images, i), show_nth_label(labels, i))

def test(images, labels):
    global cl
    cl=naivebayes(getpixels)
    train(cl, "trainingimages.txt", "traininglabels.txt")
    results = {}
    for num in cl.categories():
        results[num]={}
        results[num].setdefault("total",0)
        for init in range(0,10):
            results[num].setdefault(init,0)
    for i in range(1, 501):
        n = show_nth_label(labels, i)
        guess = cl.classify(show_nth_image(images,i), default='unknown')
        
        results[n][guess] += 1
        results[n]["total"] += 1
    
    accuracy = {}
    total_correct = 0
    total_total = 0
    for r in results:
        if results[r]["total"]!=0:
            accuracy[r] = float(results[r][r])/float(results[r]["total"])
            total_correct += results[r][r]
            total_total += results[r]["total"]
    print accuracy
    overall = float(total_correct)/float(total_total)
    print overall
    
    print " "
    print "Confusion Matrix:   Top=Actual  Vert=Guess"
    s = "      "
    print "    ",0,s,1,s,2,s,3,s,4,s,5,s,6,s,7,s,8,s,9,"  "
    for row in range(0,10):
        string = str(row)+"  "
        for col in range(0,len(results[row])-1):
            percent = round(float(results[row][col])/float(results[row]["total"]), 3)
            string+= str(percent).zfill(5) + "    "
        print string