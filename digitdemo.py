cl="cl"

from digitclass import *
from helpers import *

def d1(images, labels):
    global cl
    cl=classifier(getpixels)
    for i in range (0, 5000):
        cl.train(show_nth_image(images, i), show_nth_label(labels, i)) 
    print( cl.fcount(0, ' ') )


# def d2():
#     global cl
#     cl=classifier( getwords ) 
#     sampletrain( cl ) 
#     print( cl.fprob( 'quick', 'good' ) ) # 0.666666
#     print( cl.fprob( 'money', 'good' ) ) # 0
#     print( cl.fprob( 'money', 'bad' )  ) # 0.5


# def d3():
#     global cl
#     cl=classifier(getwords) 
#     sampletrain(cl)
#     print()
#     print( "fprob( money, good) ", cl.fprob( 'money', 'good' ))
#     print( "wprob( money, good) ",cl.weightedprob( 'money', 'good', cl.fprob )) # 0.25
#     print()
#     sampletrain(cl)
#     print()
#     print( "wprob( money, good) ", cl.weightedprob( 'money', 'good', cl.fprob)) # 0.166666666 
#     print( "fprob( money, good) ", cl.fprob( 'money', 'good' ))
 

# def d4():
#     global cl
#     cl=naivebayes(getwords)     
#     sampletrain(cl)
#     print()
#     print( 'good> ', cl.prob('quick rabbit', 'good')) # 0.15624999999999997
#     print()
#     print( 'bad> ', cl.prob('quick rabbit', 'bad')) # 0.050000000000000003 


# def d5():
#     global cl
#     cl=naivebayes(getwords) 
#     sampletrain(cl)
#     print()
#     print( '-----------')
#     print( '1> ', cl.classify('quick rabbit', default='unknown'))
#     print()
#     print( '2> ', cl.classify('quick money', default='unknown'))
#     print()
#     cl.setthreshold('bad', 3.0)
#     print( '3> ', cl.classify('quick money', default='unknown'))
#     print()
#     for i in range(10):
#         sampletrain(cl)
#         print()
#     print()
#     print( '4> ', cl.classify('quick money', default='unknown'))

