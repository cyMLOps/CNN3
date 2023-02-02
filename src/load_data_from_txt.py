from os import listdir



images = open('path_to_images.txt', 'r')
line = images.readline()
i = 0
while line and i<5 :
    image = line.rstrip()
    print(image)
    i += 1
