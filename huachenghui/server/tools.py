# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : tools.py
# @time    : 19-07-12 09:14
# @desc    : 处理视频路数获取
'''

def get_task_id():
    '''
    功能：随机获取任务ID
    return:任务ID
    '''
    import random
    import time
    task_id = random.randint(100000,999999)
    task_id = str(task_id) + str(int(time.time()*1000)%int(time.time()))
    return int(task_id)

def get_task_way_count(running_tasks):
    try:
        sum_ways = 0 #总视频路数
        for task_id in running_tasks.keys():
            sum_ways = sum_ways + len(running_tasks[task_id])
        return sum_ways
    except:
        return 0