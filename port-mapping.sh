# Assume that your GPU server is on AWS EC2 and your HTML file is on your local PC.
#chmod 600 YOUR_PEM_KEY.pem
#ssh -L 8888:localhost:8000 -i YOUR_PEM_KEY.pem ubuntu@EC2_IP
# or 
chmod 600 YOUR_PEM_KEY.pem
ssh -L 8888:localhost:8000 -N -f -i YOUR_PEM_KEY.pem ubuntu@EC2_IP