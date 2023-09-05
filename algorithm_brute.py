import pandas as pd 
import math, random, sys
from PIL import Image
from pillow_heif import register_heif_opener
import numpy as np
import scipy.cluster.hierarchy as hcluster


data = pd.read_csv("data.csv")
data.drop(data.index[1000:])

# First, let's do a brute algorithm test

# The first step is to retrieve a camera frame
register_heif_opener()
img = Image.open("sample_star.heic")


# And then find all the star pixels
threshold = 240
star_pixels = list()
for x in range(img.width):
    for y in range(img.height):
        pixel = img.getpixel((x, y))
        brightness = (pixel[0] + pixel[1] + pixel[2]) / 3
        if brightness < threshold:
            img.putpixel((x, y), 0)
        else:
            img.putpixel((x, y), (255, 255, 255))
            star_pixels.append((x, y))


# The issue is that some stars have multiple pixels
# So... how should we deal with this?
# One way of dealing with this is by using hierarchical clustering!
star_pixels = np.array(star_pixels)
threshold_2 = 1.5

clusters = hcluster.fclusterdata(star_pixels, threshold_2, criterion="distance")

star_pixel_centroids = list()
star_pixels = star_pixels.tolist()
cluster_count = dict()
index = 0
for pixel in star_pixels:
    # clusters[index] is the cluster the current pixel belongs to
    if clusters[index] not in cluster_count.keys():
        # tuple is (count, sum X, sum Y)
        cluster_count[clusters[index]] = [1, pixel[0], pixel[1]]
    else:
        cluster_count[clusters[index]][0] = cluster_count[clusters[index]][0] + 1
        cluster_count[clusters[index]][1] = cluster_count[clusters[index]][1] + pixel[0]
        cluster_count[clusters[index]][2] = cluster_count[clusters[index]][2] + pixel[1]
    index += 1


for key in cluster_count.keys():
    star_pixel_centroids.append([cluster_count[key][1]/cluster_count[key][0], cluster_count[key][2]/cluster_count[key][0]])


# Now... pick a triplet of star pixels and get the sorted set of distances between each pair of them
n1, n2, n3 = random.sample(range(0, len(star_pixel_centroids)), 3)
p1 = star_pixel_centroids[n1]
p2 = star_pixel_centroids[n2]
p3 = star_pixel_centroids[n3]

def pixel_distance(point_1, point_2):
    return ((point_1[0] - point_2[0])**2 + (point_1[1] - point_2[1])**2)**0.5

d1_2 = pixel_distance(p1, p2)
d1_3 = pixel_distance(p1, p3)
d2_3 = pixel_distance(p2, p3)

pixel_distances_set = sorted([d1_2, d1_3, d2_3])

# Now... select all possible triplets of star pixels and get the sorted distances between each pair of them
def angular_distance(dec1, dec2, ra1, ra2):
    return (math.sin(dec1) * math.sin(dec2)) + (math.cos(dec1)*math.cos(dec2)*math.cos(ra1-ra2))


def RMS(n, star_distances, pixel_distances):
    sum = 0
    for i in range(n):
        sum += ((star_distances[i] - pixel_distances[i])**2)/n
    return sum**0.5
    
min_RMS = sys.maxsize * 2 + 1
closest_star_set = None
print("Starting Brute Algo")
for index, row in data.iterrows():
    print("Current index: %s" % (index,))
    for index_2, row_2 in data.iterrows():
        #print("Current index_2: %s" % (index_2,))
        if index != index_2:
            for index_3, row_3 in data.iterrows():
                #print("Current index_3: %s" % (index_3,))
                if (index != index_3 and index_2 != index_3):
                    s1_2 = angular_distance(row['DEdeg'], row_2['DEdeg'], row['RAdeg'], row_2['RAdeg'])
                    s1_3 = angular_distance(row['DEdeg'], row_3['DEdeg'], row['RAdeg'], row_3['RAdeg'])
                    s2_3 = angular_distance(row_2['DEdeg'], row_3['DEdeg'], row_2['RAdeg'], row_3['RAdeg'])
                    star_distances_set = sorted([s1_2, s1_3, s2_3])
                    # Calculate RMS with this
                    curr_RMS = RMS(3, star_distances_set, pixel_distances_set)
                    if curr_RMS < min_RMS:
                        print("New closest found! Index %s" % (index,))
                        closest_star_set = [index, index_2, index_3]
                        min_RMS = curr_RMS

# Use this with the Yale Bright Star Catalog (only around 10,000 stars!)
def convert_RA(hours, minutes, seconds):
    return (hours + minutes/60 + seconds/3600)*15

print(min_RMS)
print(closest_star_set)
