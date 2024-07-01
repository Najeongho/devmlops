## 4-1 DevOps Hands-On

pip3 install -r requirements.txt

python3 app.py  

docker build -t app:0.1 ./

docker images

docker run -d -p 80:5050 -t app:0.1

docker ps
