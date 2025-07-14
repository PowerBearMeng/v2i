# v2i detect moving things
<!-- vscode-markdown-toc -->
* 1. [项目模块说明：](#)
	* 1.1. [real_time模块：](#real_time)
	* 1.2. [utils模块](#utils)
		* 1.2.1. [io.py](#io.py)
		* 1.2.2. [config.py](#config.py)
		* 1.2.3. [logger.py:](#logger.py:)
		* 1.2.4. [visualization.py:](#visualization.py:)
		* 1.2.5. [bev.py:](#bev.py:)
		* 1.2.6. [preprocess.py:](#preprocess.py:)
		* 1.2.7. [detection.py:](#detection.py:)
		* 1.2.8. [部分](#-1)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->


## 使用说明：
python版本为3.10.12

` conda create -n v2i python=3.10.12 `

安装项目库函数

` pip install -r requirements.txt `

运行程序：

` python main.py` 


##  1. <a name=''></a>项目模块说明：

###  1.1. <a name='real_time'></a>real_time模块：

发送端直接从ros2中监听话题/rslidar_points然后发送 
ros2zmq_send_tcp.py
ros2zmq_send_udp.py（Udp目前有问题）

### recv_real_time模块：
接收端直接监听话题

###  1.2. <a name='utils'></a>utils模块

####  1.2.1. <a name='io.py'></a>io.py 
提供点云流模拟功能，可以从文件夹中读取点云文件并模拟传感器传来的实时数据流。

####  1.2.2. <a name='config.py'></a>config.py
 包含项目的各项配置参数，如点云处理的区域范围（ROI）、地面模型配置、动态检测参数及可视化配置。

####  1.2.3. <a name='logger.py:'></a>logger.py:
提供性能日志记录功能，可将帧处理时间、动态点数量等性能数据记录到CSV文件中。

####  1.2.4. <a name='visualization.py:'></a>visualization.py:
提供实时可视化功能，用于渲染并动态更新点云和几何对象。

####  1.2.5. <a name='bev.py:'></a>bev.py: 
实现鸟瞰图（BEV）的生成与运动检测功能，可将3D点云转换为2D视图并通过帧间差异检测动态物体。

####  1.2.6. <a name='preprocess.py:'></a>preprocess.py: 
封装点云预处理功能，包括地面模型计算、移除地面点、过滤指定范围内的点等。

####  1.2.7. <a name='detection.py:'></a>detection.py:
提供动态物体检测功能，支持DBSCAN聚类及基于区域增长的动态点检测。

####  1.2.8. <a name='-1'></a>部分
calculate.py和dbscan.py后续没有使用 没有说明

### main函数
#### main.py
检测原始的移动点云 最初的版本
#### main_box.py
含物体运动框的检测
#### main_bev.py
针对BEV的检测