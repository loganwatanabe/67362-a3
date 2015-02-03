img_height = 28
img_width = 28

def show_nth_image(fname, n):
    f = open(fname, "rU")
    img = []
    # skip first n-1 images
    # f.seek( (n-1)*img_height )
    for i in range( (n-1)*img_height):
        f.readline()
    for i in range(img_height):
        line = f.readline().rstrip('\n')
        img.append(line)
    return img

def show_nth_label(fname, n):
    f = open(fname, "rU")
    # skip first n-1 images
    # f.seek( (n-1)*img_height )
    for i in range(n-1):
        f.readline()
    label = int(f.readline().rstrip('\n'))
    return label