# environment  
ubuntu 16.04/18.04  
python==3.6  
certifi==2016.2.28  
cffi==1.10.0  
numpy==1.16.4  
olefile==0.44  
opencv-python==4.1.0.25  
Pillow==4.2.1  
pycparser==2.18  
scipy==1.3.0  
six==1.10.0  
torch==1.1.0  
torchvision==0.3.0  
psutil  
nvidia-ml-py  
django==2.1.10  
djangorestframework  
django-filter  
pip install django-cors-headers  

创建项目  
创建django项目：django-admin.py startproject video_analysis  

查看开放的端口  
netstat -nupl 查看udp协议的端口号  
netstat -ntpl 查看tcp协议的端口号

检查端口是否开放  
su root
lsof -i:8080 没有任何输出 则未开放端口
开放端口
sudo iptables -I INPUT -p tcp --dport 8080 -j ACCEPT
sudo iptables-save
以上方法只能临时开启端口，服务器重启后则消失
永久开启端口：
安装iptables-persistent
sudo apt-get install iptables-persistent

持久化规则
sudo netfilter-persistent save
sudo netfilter-persistent reload

启动项目：  
python manage.py runserver 0.0.0.0:8097  
浏览器输入：http://127.0.0.1:8087

创建app(这里的app是指项目中的每个功能。如用户登陆注册可以放在同一个app中)  
django-admin startapp server  

http://127.0.0.1:8000/index_form/  
释放占用的端口：  
sudo fuser -k 8080/tcp  
