import sys
import pylab as pl

# if len(sys.argv)!=3:
# 	print('usage: python map.py $PREDICT_FILE $GROUND_TRUTH_FILE\n')
# 	exit()

predict_file = 'val_result_ensemble.txt'
ground_truth_file = 'val_gt.txt'

predict_dict = dict()
ground_truth_dict = dict()

def get_info(info_file, info_dict):
	with open(info_file, 'r') as f:
		lines = f.readlines()
	i = 0
	while i < len(lines):
		if '/' in lines[i]:
			im_id = lines[i].rstrip()
			num = int(lines[i + 1].rstrip())
			info_dict[im_id] = list()
			info_dict[im_id].append(num)
			i = i + 2
			for _ in range(num):
				x1, y1, w, h, conf = map(float, lines[i].rstrip().split(' '))
				x2 = x1 + w
				y2 = y1 + h
				info_dict[im_id].append([x1, y1, x2, y2, conf])
				i = i + 1
			
		else:
			i = i + 1
	bbox_num = 0
	for key in info_dict.keys():
		bbox_num += info_dict[key][0]
	return bbox_num

predict_bbox_num = get_info(predict_file, predict_dict)
ground_truth_bbox_num = get_info(ground_truth_file, ground_truth_dict)
score_list = list() # score
match_list = list() # match

def iou(predict_bbox, ground_truth_bbox):
	'''
	:param predict_bbox: list, format,[x1,y1,x2,y2,conf]
	:param ground_truth_bbox: list, format, [x1,y1,x2,y2,conf]
	:return: value,area of area
	'''
	predict_area = (predict_bbox[2] - predict_bbox[0])*(predict_bbox[3] - predict_bbox[1])
	ground_truth_area = (ground_truth_bbox[2] - ground_truth_bbox[0])*(ground_truth_bbox[3] - ground_truth_bbox[1])
	inter_x = min(predict_bbox[2],ground_truth_bbox[2]) - max(predict_bbox[0],ground_truth_bbox[0])
	inter_y = min(predict_bbox[3],ground_truth_bbox[3]) - max(predict_bbox[1],ground_truth_bbox[1])
	if inter_x<=0 or inter_y<=0:
		return 0
	inter_area = inter_x*inter_y
	return inter_area / (predict_area+ground_truth_area-inter_area)

def compare(predict_list, ground_truth_list, score_list, match_list):
	'''
	:param predict_list: list, format (x,y,w,h,c)
	:param ground_truth_list: list, format (x,y,w,h,c)
	:param score_list:list, store confidence matched
	:param match_list:list, stroe the num of matched boxes
	:return: None
	'''
	ground_truth_unuse = [True for i in range(1, len(ground_truth_list))]
	# for predict_bbox in predict_list:
	for j in range(1, len(predict_list)):
		# print('j={}'.format(j))
		predict_bbox = predict_list[j]
		match = False
		for i in range(1, len(ground_truth_list)):
			# print('i={}'.format(i))
			if ground_truth_unuse[i-1]:
				# 只寻找一个匹配框,并且是第一个满足条件的,遍历的方式比较慢
				if iou(predict_bbox, ground_truth_list[i])>0.5: # 阈值=0.5
					match = True
					ground_truth_unuse[i-1] = False
					break
		score_list.append(predict_bbox[-1])
		match_list.append(int(match))


print('compare...')
for key in predict_dict.keys():
	compare(predict_dict[key], ground_truth_dict[key], score_list, match_list)


# zip()传入可迭代的对象,返回一个以元组为元素的列表,拼接的过程中会以最短的长度截断
score_match_list = list(zip(score_list, match_list)) # [(score,match),(score,match),...]
score_match_list.sort(key = lambda x:x[0], reverse = True)
print('calculate precision/recall...')

p = list()
r = list()
predict_num = 0
truth_num = 0

for item in score_match_list:
	predict_num+=1
	truth_num+=item[1]
	# recall = TP/(TP + FN)
	r.append(float(truth_num)/ground_truth_bbox_num)
	# precision = TP/(TP + TN)
	p.append(float(truth_num)/predict_num)
mAP = 0
for i in range(1,len(p)):
	# calculate mean Average precision
	mAP += (p[i-1]+p[i])/2*(r[i]-r[i-1])
print('mAP:{}'.format(mAP))
pl.plot(r,p)
pl.show()
