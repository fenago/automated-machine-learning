#### Local Setup Guide

**Using docker**

1. Install Docker: https://docs.docker.com/engine/install/
2. Pull Docker Image: `docker pull fenago/automated-machine-learning`
3. Start Lab Environment: 

`docker run -d --restart=always --user root -p 80:80 --name automated -e GRANT_SUDO=yes -e JUPYTER_ENABLE_LAB=yes fenago/automated-machine-learning jupyter lab --port 80 --allow-root`

Open `http://localhost:80` or `http://<DOCKER_HOST_IP>:80` in browser and login with `1234` password.


**Optional:**

* Restart Lab Environment: 

`docker restart automated`

* Delete Lab Environment: 

```
docker stop automated
docker rm automated
```
