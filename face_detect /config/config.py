#硬件服务器件参数配置文件
server={
    "GPU":0.81, #GPU的使用率阈值
    "GPU_cache":0.82, #GPU_cache的使用率阈值
    "CPU":0.83, #CPU的使用率阈值
    "Memory":0.84, #内存的使用率阈值
    "video_routes":16, #并发处理视频路数
    #视频分析服务器的信息
    "IP":"192.168.2.203",
    "PORT":"8080"
}

#redis参数配置
redis={
    "host":"127.0.0.1",
    "port":"6379",
    "password":"123456"
}

#启用的算法类型:AlignedReid(表示每隔一秒进行一次特征提取),daSiamRPN(采用daSiamRPN行人跟踪算法),eanet,bagReid
ALGORITHM_TYPE="bagReid"
#支持最大视频流路数
video_routes=16

bagReid={
    #欧氏距离阈值
    "dist_threshold":1.40,
    "MODEL_PATH":'',
    #是否播放视频
    "IS_SHOW":'0'
}
# 0.70 0.532 0.836
# 0.72 0.581 0.811
# 0.74 0.622 0.786
# 0.76 0.667 0.750
# 0.78 0.700 0.711
# 0.80 0.741 0.681
# 0.82 0.781 0.634
# 0.84 0.822 0.582
# 0.86 0.848 0.533
# 0.88 0.873 0.478
# 0.90 0.895 0.427

eanet={

}

alignedReid={
    #欧氏距离阈值
    "EUCLID_DISTANCE_THRESHOLD1":0.53,
    #是否启用历史行人辅助追踪功能(1表示启动，其他表示不启用)
    "HISTORY_ASSIST_TRACK":0,
    #行人检测API
    "MODEL_TYPE":"yolo",
    "MODEL_PATH":'',
    #是否播放视频
    "IS_SHOW":0,
    #提取行人特征的之后，是否将图片左右翻转(1代表是，其他代表否)
    "FLIPHOR_FLAG":1
}

FLIPHOR_FLAG=0

