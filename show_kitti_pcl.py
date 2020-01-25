import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

def readXYZfile():
	filename = "/home/lyh/lab/dataset/KITTI_RAW/2011_09_30_drive_0020_sync/2011_09_30/2011_09_30_drive_0020_sync/velodyne_points/data/0000000450.bin"
	pointcloud_all = np.fromfile(filename, dtype=np.float32, count=-1).reshape([-1,4])
	
	pointcloud = np.zeros((4096,3), dtype=np.float)
	for i in range(4096):
		rand = random.randint(0,pointcloud_all.shape[0]-1)
		pointcloud[i,:] = pointcloud_all[rand,0:3]
	
	for i in range(pointcloud_all.shape[0]):
		print(pointcloud_all[i,0],pointcloud_all[i,1],pointcloud_all[i,2])
		
	np.savetxt('result.txt', pointcloud_all+50000, fmt = "%.5f",delimiter = ',')
	exit()
	print(pointcloud.shape)
	x = pointcloud[:, 0]  # x position of point
	y = pointcloud[:, 1]  # y position of point
	z = pointcloud[:, 2]  # z position of point
	point = [x,y,z]
	
	return point
 

def displayPoint(data,title):
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    while len(data[0]) > 200000:
    	print("too much points")
    	exit()
    
    fig=plt.figure() 
    ax=Axes3D(fig) 
    ax.set_title(title) 
    ax.scatter3D(data[0], data[1],data[2], c = 'r', marker = '.') 
    ax.set_xlabel('x') 
    ax.set_ylabel('y') 
    ax.set_zlabel('z') 
    plt.show()

if __name__ == "__main__":
	data = readXYZfile()
	displayPoint(data, "rabbit")