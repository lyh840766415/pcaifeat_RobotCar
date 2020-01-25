import numpy as np
from loading_input import *
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from multiprocessing.dummy import Pool as ThreadPool

Database_KDtree = []
Dataset_size = 0
cnt = 0

def kdtree_query(feat):
	global cnt
	dist, ind = Database_KDtree.query([feat], k=round(Dataset_size/4),sort_results = True)	
	cnt = cnt + 1
	print("kdtree query ",cnt)
	return ind

def main():
	global Database_KDtree,Dataset_size
	
	print("loading feature ......")
	all_feat = np.loadtxt('model_378000_all_feat')
	#all_feat = all_feat[0:3000]
	all_feat_list = all_feat.tolist()
	Dataset_size = all_feat.shape[0]
	print(Dataset_size," Dataset_size")
	#for feat in all_feat_list:
	#	print(feat)
	#exit()
	print("feature loaded.\nloading label ......")
	label = get_queries_dict('generate_queries/pcai_training.pickle')
	print("label loaded")
	
	
	Database_KDtree = KDTree(all_feat)
	pool = ThreadPool(100)
	ind = pool.map(kdtree_query,all_feat_list)
	pool.close()
	pool.join()
	
	recall = np.zeros([Dataset_size,26],dtype = np.float32)
	
	for i in range(Dataset_size):
		positive_i = np.array(label[i]['positives'])
		for j in range(25):
			retrieved = ind[i][0,0:round(1000/(100/(j+1)))]
			inter = np.intersect1d(positive_i,retrieved)
			recall[i,j]=inter.shape[0]/positive_i.shape[0]
			
		#top1 acc	
		retrieved = ind[i][0,1:2]
		inter = np.intersect1d(positive_i,retrieved)
		recall[i,25] = inter.shape[0]/1;
		
			
	
	print(recall)	
	ave_recall = recall.mean(axis = 0)[0:25]
	ave_top1_acc = recall.mean(axis = 0)[25]
	for i in range(ave_recall.shape[0]):
		print(i+1,ave_recall[i])
		
	print("top1_acc = ",ave_top1_acc)
	
	#print("sample %d top at %d reall = %f"%(i,25,recall[i,24]))
	
	exit()			
		
	
	print(len(ind))
	print(label[0]['positives'])
	a = np.array(label[0]['positives'])
	b = ind[0]
	print(a)
	print(b)
	inter = np.intersect1d(a,b)
	print(inter)
	recall = inter.shape[0]/a.shape[0]
	print("recall = ",recall)
	
	
	
	

if __name__ == '__main__':
	main()