from loading_input import *
from pointnetvlad.pointnetvlad_cls import *
import nets.resnet_v1_50 as resnet
import tensorflow as tf
import cv2
from time import *

#1. load test file and label
#2. init network and calculate all feature
#3. compute recall according to feature and label


TRAIN_FILE = 'generate_queries/pcai_training.pickle'
TRAINING_QUERIES = get_queries_dict(TRAIN_FILE)
BATCH_SIZE = 200
EMBBED_SIZE = 128
MODEL_PATH = '/home/lyh/lab/pcaifeat/log/train_save/model_378000.ckpt'

#ready to delete
def get_learning_rate(epoch):
	learning_rate = BASE_LEARNING_RATE*((0.9)**(epoch//5))
	learning_rate = tf.maximum(learning_rate, 0.00001)
	return learning_rate
	
def get_bn_decay(batch):
	#batch norm parameter
	DECAY_STEP = 20000
	BN_INIT_DECAY = 0.5
	BN_DECAY_DECAY_RATE = 0.5
	BN_DECAY_DECAY_STEP = float(DECAY_STEP)
	BN_DECAY_CLIP = 0.99
	bn_momentum = tf.train.exponential_decay(BN_INIT_DECAY,batch*BATCH_SIZE,BN_DECAY_DECAY_STEP,BN_DECAY_DECAY_RATE,staircase=True)
	bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
	return bn_decay

def init_imgnetwork():
	images_placeholder = tf.placeholder(tf.float32,shape=[BATCH_SIZE,144,288,3])
	EMBBED_SIZE = 128
	endpoints,body_prefix = resnet.endpoints(images_placeholder,is_training=False)
	#fix output feature to 128-d
	img_feat = tf.layers.dense(endpoints['model_output'], EMBBED_SIZE)
	
	return images_placeholder,img_feat
	
def init_pcnetwork(batch):
	pc_placeholder = tf.placeholder(tf.float32,shape=[BATCH_SIZE,4096,3])
	is_training_pl = tf.Variable(False, name = 'is_training')
	bn_decay = get_bn_decay(batch)
	endpoints = pointnetvlad(pc_placeholder,is_training_pl,bn_decay)
	pc_feat = tf.layers.dense(endpoints,EMBBED_SIZE)
	
	return pc_placeholder,pc_feat
	
def init_pcainetwork():
	batch = tf.Variable(0)
	epoch_num_placeholder = tf.placeholder(tf.float32, shape=())
	images_placeholder,img_feat = init_imgnetwork()
	pc_placeholder,pc_feat = init_pcnetwork(batch)
	img_pc_concat_feat = tf.concat((pc_feat,img_feat),axis=1)
	pcai_feat = tf.layers.dense(img_pc_concat_feat,256)
	
	ops = {
		'images_placeholder':images_placeholder,
		'pc_placeholder':pc_placeholder,
		'epoch_num_placeholder':epoch_num_placeholder,
		'batch':batch,
		'pcai_feat':pcai_feat}
	
	return ops
	
def cal_all_features(ops,sess):
	print("pickle_size = ",len(TRAINING_QUERIES.keys()))
	train_file_idxs = np.arange(0,len(TRAINING_QUERIES.keys()))
	all_feat = np.empty([0,256],dtype=np.float32)
	print(all_feat.shape)
	for i in range(len(train_file_idxs)//BATCH_SIZE):
		batch_keys = train_file_idxs[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
		pc_files=[]
		img_files=[]
		if i<0:
			continue
		
		for j in range(BATCH_SIZE):
			pc_files.append(TRAINING_QUERIES[batch_keys[j]]["query_pc"])
			img_files.append(TRAINING_QUERIES[batch_keys[j]]["query_img"])
			#print(batch_keys[j])
		
		begin_time = time()
		
		pc_data,_ = load_pc_files(pc_files)
		img_data,_ = load_images(img_files)
		
		end_time = time()
		
		print ('load time ',end_time - begin_time)
		'''
		print(pc_data.shape)
		print(img_data.shape)
		print(batch_keys[BATCH_SIZE-1])
		'''
		#for j in range(4096):
		#	print(pc_data[10][j][0],pc_data[10][j][1],pc_data[10][j][2])
		#cv2.imshow("test",img_data[10])
		#cv2.waitKey(0)
		#exit()
		
		train_feed_dict = {
			ops['images_placeholder']:img_data,
			ops['pc_placeholder']:pc_data,
			ops['epoch_num_placeholder']:0,
		}
		
		begin_time = time()
		pcai_feat,_ = sess.run([ops['pcai_feat'],ops['batch']],feed_dict=train_feed_dict)
		end_time = time()
		print ('feature time ',end_time - begin_time)
		
		all_feat = np.concatenate((all_feat,pcai_feat),axis=0)

		print(all_feat.shape)
		
	#hold edge case
	remind_index = len(train_file_idxs)%BATCH_SIZE
	tot_batches = len(train_file_idxs)//BATCH_SIZE
	pc_files=[]
	img_files=[]		
	batch_keys = train_file_idxs[tot_batches*BATCH_SIZE:tot_batches*BATCH_SIZE+remind_index]
	
		
	for i in range(BATCH_SIZE):
		cur_index = min(remind_index-1,i)
		pc_files.append(TRAINING_QUERIES[batch_keys[cur_index]]["query_pc"])
		img_files.append(TRAINING_QUERIES[batch_keys[cur_index]]["query_img"])
		#print(batch_keys[cur_index])
	
	pc_data,_ = load_pc_files(pc_files)
	img_data,_ = load_images(img_files)
	
	'''
	print(pc_data.shape)
	print(img_data.shape)
	print(batch_keys[cur_index])
	'''
	
	train_feed_dict = {
		ops['images_placeholder']:img_data,
		ops['pc_placeholder']:pc_data,
		ops['epoch_num_placeholder']:0,
	}
	pcai_feat,_ = sess.run([ops['pcai_feat'],ops['batch']],feed_dict=train_feed_dict)
	all_feat = np.concatenate((all_feat,pcai_feat),axis=0)
	print(all_feat.shape)
	all_feat = all_feat[0:len(TRAINING_QUERIES.keys()),:]
	print(all_feat.shape)
	np.savetxt("model_378000_all_feat",all_feat)
	
	
def main():
	ops = init_pcainetwork()
	print("network initialized")
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	saver = tf.train.Saver()
	sess = tf.Session(config=config)
	
	saver.restore(sess, MODEL_PATH)
	print("model restored")
	
	#print(TRAINING_QUERIES[0])
	
	cal_all_features(ops,sess)
	
	
if __name__ == '__main__':
	main()