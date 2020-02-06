import numpy as np
from loading_input import *
from pointnetvlad.pointnetvlad_cls import *
import nets.resnet_v1_50 as resnet
import tensorflow as tf
from time import *
import pickle


DATABASE_FILE= 'generate_queries/RobotCar_oxford_evaluation_database.pickle'
QUERY_FILE= 'generate_queries/RobotCar_oxford_evaluation_query.pickle'
PC_IMG_MATCH_FILE = 'generate_queries/pcai_pointcloud_image_match_test.pickle'
IMAGE_PATH = '/data/lyh/RobotCar'
BATCH_SIZE = 60
EMBBED_SIZE = 256
PCAI_EMBBED_SIZE = 256

MODEL_PATH = "/home/lyh/lab/pcaifeat_RobotCar/model/256_dim_model_01410000"
MODEL_NAME = "model_01410000.ckpt"

DATABASE_SETS= get_sets_dict(DATABASE_FILE)
QUERY_SETS= get_sets_dict(QUERY_FILE)
PC_IMG_MATCH_DICT = get_pc_img_match_dict(PC_IMG_MATCH_FILE)

global DATABASE_VECTORS
DATABASE_VECTORS=[]

global QUERY_VECTORS
QUERY_VECTORS=[]

def get_correspond_img(pc_filename):
	timestamp = pc_filename[-20:-4]
	seq_name = pc_filename[-55:-36]
	#print("Portental Error")
	#print(timestamp)
	#print(seq_name)
	#print(PC_IMG_MATCH_DICT[seq_name][timestamp])
	image_ind = PC_IMG_MATCH_DICT[seq_name][timestamp]
	image_timestamp = image_ind[random.randint(0,len(image_ind)-1)]
	image_filename = os.path.join(IMAGE_PATH,seq_name,"stereo/centre","%s.png"%(image_timestamp))
	#print(image_filename)
	#if os.path.exists(image_filename):
	#	print("exist")
	return image_filename


def get_latent_vectors(sess,ops,dict_to_process):
	print("dict_size = ",len(dict_to_process.keys()))
	train_file_idxs = np.arange(0,len(dict_to_process.keys()))
	all_feat = np.empty([0,PCAI_EMBBED_SIZE],dtype=np.float32)
	
	
	for i in range(len(train_file_idxs)//BATCH_SIZE):
		batch_keys = train_file_idxs[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
		pc_files=[]
		img_files=[]
		if i<0:
			print("Error, ready for delete")
			continue
		
		for j in range(BATCH_SIZE):
			pc_files.append(dict_to_process[batch_keys[j]]["query"])			
			img_files.append(get_correspond_img(dict_to_process[batch_keys[j]]["query"]))
			#print(batch_keys[j])
		
		'''
		print(pc_files[0])
		print(img_files[0])
		if os.path.exists(pc_files[0]):
			print("exist",pc_files[0])
		if os.path.exists(img_files[0]):
			print("exist",img_files[0])
		exit()
		'''
		
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
		
	#no edge case
	if len(train_file_idxs)%BATCH_SIZE == 0:
		return all_feat
			
	#hold edge case
	remind_index = len(train_file_idxs)%BATCH_SIZE
	tot_batches = len(train_file_idxs)//BATCH_SIZE
	pc_files=[]
	img_files=[]		
	batch_keys = train_file_idxs[tot_batches*BATCH_SIZE:tot_batches*BATCH_SIZE+remind_index]
	
		
	for i in range(BATCH_SIZE):
		cur_index = min(remind_index-1,i)
		pc_files.append(dict_to_process[batch_keys[cur_index]]["query"])			
		img_files.append(get_correspond_img(dict_to_process[batch_keys[cur_index]]["query"]))
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
	all_feat = all_feat[0:len(dict_to_process.keys()),:]
	return all_feat
	
def output_to_file(output, filename):
	with open(filename, 'wb') as handle:
		pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print("Done ", filename)
	
	
def cal_all_features(ops,sess):
	database_feat = []
	query_feat = []
	
	for i in range(len(DATABASE_SETS)):
		cur_feat = get_latent_vectors(sess, ops, DATABASE_SETS[i])
		database_feat.append(cur_feat)
	for j in range(len(QUERY_SETS)):
		cur_feat = get_latent_vectors(sess, ops, QUERY_SETS[j])
		query_feat.append(cur_feat)
	
	output_to_file(database_feat,"database_feat_"+MODEL_NAME[0:-5]+".pickle")
	output_to_file(query_feat,"query_feat_"+MODEL_NAME[0:-5]+".pickle")
		
		
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
	endpoints,body_prefix = resnet.endpoints(images_placeholder,is_training=True)
	
	return images_placeholder,endpoints['model_output']


def init_pcnetwork(batch):
	pc_placeholder = tf.placeholder(tf.float32,shape=[BATCH_SIZE,4096,3])
	is_training_pl = tf.Variable(True, name = 'is_training')
	bn_decay = get_bn_decay(batch)
	endpoints = pointnetvlad(pc_placeholder,is_training_pl,bn_decay)

	return pc_placeholder,endpoints


def init_pcainetwork():
	batch = tf.Variable(0)
	epoch_num_placeholder = tf.placeholder(tf.float32, shape=())
	#with tf.variable_scope("imgnet_var"):
	images_placeholder,img_feat_ori = init_imgnetwork()
	#with tf.variable_scope("pcnet_var"):
	pc_placeholder,pc_feat_ori = init_pcnetwork(batch)
	
	with tf.variable_scope("fusion_var"):
		img_feat = tf.layers.dense(img_feat_ori, EMBBED_SIZE)
		pc_feat = tf.layers.dense(pc_feat_ori, EMBBED_SIZE)
		img_pc_concat_feat = tf.concat((pc_feat,img_feat),axis=1)
		pcai_feat = tf.layers.dense(img_pc_concat_feat,PCAI_EMBBED_SIZE)
		
	ops = {
		'images_placeholder':images_placeholder,
		'pc_placeholder':pc_placeholder,
		'epoch_num_placeholder':epoch_num_placeholder,
		'batch':batch,
		'pcai_feat':pcai_feat}
	return ops



def main():
	ops = init_pcainetwork()
	print("network initialized")
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	saver = tf.train.Saver()
	sess = tf.Session(config=config)
	
	saver.restore(sess, os.path.join(MODEL_PATH,MODEL_NAME))
	print("model restored")
	cal_all_features(ops,sess)


if __name__ == "__main__":
	main()
