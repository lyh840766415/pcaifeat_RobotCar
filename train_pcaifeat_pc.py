import numpy as np
from loading_input import *
from pointnetvlad.pointnetvlad_cls import *
import random
import cv2
import nets.resnet_v1_50 as resnet
import tensorflow as tf

LOG_DIR = "log"
TRAIN_FILE = 'generate_queries/pcai_training.pickle'
TRAINING_QUERIES = get_queries_dict(TRAIN_FILE)
BATCH_NUM_QUERIES = 2
EPOCH = 10
POSITIVES_PER_QUERY = 2
NEGATIVES_PER_QUERY = 2
EMBBED_SIZE = 128
BASE_LEARNING_RATE = 1e-5

#learning rate halfed every 5 epoch
def get_learning_rate(epoch):
	learning_rate = BASE_LEARNING_RATE*((0.9)**(epoch//5))
	learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
	return learning_rate

def get_bn_decay(batch):
	#batch norm parameter
	DECAY_STEP = 20000
	BN_INIT_DECAY = 0.5
	BN_DECAY_DECAY_RATE = 0.5
	BN_DECAY_DECAY_STEP = float(DECAY_STEP)
	BN_DECAY_CLIP = 0.99
	bn_momentum = tf.train.exponential_decay(BN_INIT_DECAY,batch*BATCH_NUM_QUERIES,BN_DECAY_DECAY_STEP,BN_DECAY_DECAY_RATE,staircase=True)
	bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
	return bn_decay

#module that uesed to extract image feature
#input
	#image data
#output
	#image feature
def init_imgnetwork():
	images_placeholder = tf.placeholder(tf.float32,shape=[None,144,288,3])
	EMBBED_SIZE = 128
	endpoints,body_prefix = resnet.endpoints(images_placeholder,is_training=True)
	#fix output feature to 128-d
	img_feat = tf.layers.dense(endpoints['model_output'], EMBBED_SIZE)
	return images_placeholder,img_feat


#module that uesed to extract point cloud feature
#input
	#point cloud data
#output
	#point cloud feature
def init_pcnetwork(batch):
	BATCH_NUM_QUERIES = 2
	pc_placeholder = tf.placeholder(tf.float32,shape=[BATCH_NUM_QUERIES*(1+POSITIVES_PER_QUERY+NEGATIVES_PER_QUERY),4096,3])
	is_training_pl = tf.Variable(True, name = 'is_training')
	#bn_decay = tf.Variable(1.0,name = 'bn_decay')
	bn_decay = get_bn_decay(batch)
	#bn_decay = get_bn_decay(100)

	print(bn_decay)

	endpoints = pointnetvlad(pc_placeholder,is_training_pl,bn_decay)
	pc_feat = tf.layers.dense(endpoints,EMBBED_SIZE)
	return pc_placeholder,pc_feat



#module that used to init network
#input:
	#image_placeholder
	#pointcloud_placeholder

#output
	#image feature
	#pointcloud feature
	#combine feature
	#losses
	#training_ops

def init_pcainetwork():
	batch = tf.Variable(0)
	epoch_num_placeholder = tf.placeholder(tf.float32, shape=())
	images_placeholder,img_feat = init_imgnetwork()
	pc_placeholder,pc_feat = init_pcnetwork(batch)
	img_feat = tf.reshape(img_feat,[BATCH_NUM_QUERIES,(1+POSITIVES_PER_QUERY+NEGATIVES_PER_QUERY),img_feat.shape[1]])
	pc_feat = tf.reshape(pc_feat,[BATCH_NUM_QUERIES,(1+POSITIVES_PER_QUERY+NEGATIVES_PER_QUERY),pc_feat.shape[1]])
	q_img_vec, pos_img_vec, neg_img_vec = tf.split(img_feat, [1,POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY],1)
	q_pc_vec, pos_pc_vec, neg_pc_vec = tf.split(pc_feat, [1,POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY],1)
	q_vec = tf.concat((q_img_vec,q_pc_vec),axis=2)
	pos_vec = tf.concat((pos_img_vec,pos_pc_vec),axis=2)
	neg_vec = tf.concat((pos_img_vec,pos_pc_vec),axis=2)


	img_loss = triplet_loss(q_img_vec, pos_img_vec, neg_img_vec, 0.5)
	pc_loss = triplet_loss(q_pc_vec, pos_pc_vec, neg_pc_vec, 0.5)
	
	all_loss = triplet_loss(q_vec, pos_vec, neg_vec, 0.5)
	
	tf.summary.scalar('img_loss', img_loss)
	tf.summary.scalar('pc_loss', pc_loss)
	tf.summary.scalar('all_loss', all_loss)
	
	# Get training operator
	learning_rate = get_learning_rate(epoch_num_placeholder)
	tf.summary.scalar('learning_rate', learning_rate)
	optimizer = tf.train.AdamOptimizer(learning_rate)
	
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	
	with tf.control_dependencies(update_ops):
		train_op = optimizer.minimize(pc_loss, global_step=batch)
	
	# Add ops to save and restore all the variables.
	saver = tf.train.Saver()
	print(q_img_vec)
	print(pos_img_vec)
	print(neg_img_vec)

	print(q_pc_vec)
	print(pos_pc_vec)
	print(neg_pc_vec)
	
	print(q_vec)
	print(pos_vec)
	print(neg_vec)
	# Add summary writers
	merged = tf.summary.merge_all()

	return images_placeholder,pc_placeholder,epoch_num_placeholder,img_feat,pc_feat,all_loss,img_loss,pc_loss,train_op,merged,batch

#module that used to load data from Hard Disk
#input
	#data information in the Hard Disk

#output
	#numpy matrix in the memory
def get_query_tuple(dict_value, num_pos, num_neg, QUERY_DICT):
	query_pc,success_1_pc=load_pc_file(dict_value["query_pc"]) #Nx3
	query_img,success_1_img = load_image(dict_value["query_img"])

	random.shuffle(dict_value["positives"])
	pos_pc_files=[]
	pos_img_files=[]
	#load positive pointcloud
	for i in range(num_pos):
		pos_pc_files.append(QUERY_DICT[dict_value["positives"][i]]["query_pc"])
		pos_img_files.append(QUERY_DICT[dict_value["positives"][i]]["query_img"])

	#positives= load_pc_files(dict_value["positives"][0:num_pos])
	positives_pc,success_2_pc=load_pc_files(pos_pc_files)
	positives_img,success_2_img=load_images(pos_img_files)

	neg_pc_files=[]
	neg_img_files=[]
	neg_indices=[]
	random.shuffle(dict_value["negatives"])
	for i in range(num_neg):
		neg_pc_files.append(QUERY_DICT[dict_value["negatives"][i]]["query_pc"])
		neg_img_files.append(QUERY_DICT[dict_value["negatives"][i]]["query_img"])
		neg_indices.append(dict_value["negatives"][i])

	negatives_pc,success_3_pc=load_pc_files(neg_pc_files)
	negatives_img,success_3_img=load_images(neg_img_files)


	if(success_1_pc and success_1_img and success_2_pc and success_2_img and success_3_pc and success_3_img):
		query_pc = np.expand_dims(query_pc,axis = 0)
		query_img = np.expand_dims(query_img,axis = 0)
		img = np.concatenate((query_img,positives_img,negatives_img),axis=0)
		pc = np.concatenate((query_pc,positives_pc,negatives_pc),axis=0)

		return [pc,img],True


	return [query_pc,query_img,positives_pc,positives_img,negatives_pc,negatives_img],False

#module that pass the batch_data to tensorflow placeholder
def training_one_batch():
	return

#module that log the training result and evaluate the performance
def evalute_and_log():
	return



def main():
	images_placeholder,pc_placeholder,epoch_num_placeholder,img_feat,pc_feat,all_loss,img_loss,pc_loss,train_op,merged,batch = init_pcainetwork()
	print(TRAINING_QUERIES[0])
	error_cnt = 0

	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	saver = tf.train.Saver()

	#Start training
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'pc_train'),sess.graph)
		for ep in range(EPOCH):
			train_file_idxs = np.arange(0,len(TRAINING_QUERIES.keys()))
			#print(train_file_idxs)
			np.random.shuffle(train_file_idxs)
			#print(train_file_idxs)
			print('train_file_num = %f , BATCH_NUM_QUERIES = %f , iteration per batch = %f' %(len(train_file_idxs), BATCH_NUM_QUERIES,len(train_file_idxs)//BATCH_NUM_QUERIES))

			for i in range(len(train_file_idxs)//BATCH_NUM_QUERIES):
				batch_keys= train_file_idxs[i*BATCH_NUM_QUERIES:(i+1)*BATCH_NUM_QUERIES]
				#used to filter error data
				faulty_tuple = False
				#used to save training data
				q_tuples = []
				for j in range(BATCH_NUM_QUERIES):
					#determine whether positive & negative is enough
					if len(TRAINING_QUERIES[batch_keys[j]]["negatives"]) < NEGATIVES_PER_QUERY:
						print("Error Negative is not enough")
						faulty_tuple = True
						break
					if len(TRAINING_QUERIES[batch_keys[j]]["positives"]) < POSITIVES_PER_QUERY:
						print("Error Positive is not enough")
						faulty_tuple = True
						break


					cur_tuples,success= get_query_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES)
					if not success:
						faulty_tuple = True
						break

					q_tuples.append(cur_tuples)

				if faulty_tuple:
					error_cnt += 1
					continue

				cur_bat_pc = q_tuples[0][0]
				cur_bat_img = q_tuples[0][1]

				for bat, cur_tuple in enumerate(q_tuples):
					if bat == 0:
						continue
					cur_bat_pc = np.concatenate((cur_bat_pc,cur_tuple[0]),axis=0)
					cur_bat_img = np.concatenate((cur_bat_img,cur_tuple[1]),axis=0)

				#print(cur_bat_pc.shape)
				#print(cur_bat_img.shape)


				#start training
				train_feed_dict = {
					images_placeholder:cur_bat_img,
					pc_placeholder:cur_bat_pc,
					epoch_num_placeholder:ep
				}

				summary,step,run_loss,_,batch_num= sess.run([merged,batch,img_loss,train_op,batch],feed_dict = train_feed_dict)
				
				train_writer.add_summary(summary, step)

				#print("image feat",image_feat.shape)
				#print("image feat",point_feat.shape)
				
				print("batch_num = %d , loss = %f"%(batch_num,run_loss))

	print("error_cnt = %d"%(error_cnt))

			#training_one_batch()

			#evaluate_and_log()

if __name__ == '__main__':
	main()