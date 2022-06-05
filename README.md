# Super Resolution Model API
[![Language](https://img.shields.io/badge/language-Python-blue.svg?style=flat)](https://www.python.org)
[![Framework](https://img.shields.io/badge/Pytorch-red.svg?style=flat&logo=pytorch&logoColor=white)](http://www.pytorch.org/news.html)
[![Framework](https://img.shields.io/badge/Fastapi-darkgreen.svg?style=flat&logo=fastapi&logoColor=white)](http://www.fastapi.org/news.html)
![hosted](https://img.shields.io/badge/Heroku-430098?style=flat&logo=heroku&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-blue?style=flat&logo=docker&logoColor=white)
![build](https://img.shields.io/badge/build-passing-brightgreen.svg?style=flat)

An API used to upscale the resolution of a given input image. Built with Pytorch, FastAPI and Docker.



## Problem Statement
The aim of this project was to upscale the resolution of images using deep learning techniques and algorithms such as GANs and also deploy the model as an API using FastAPI framework. The model was developed using Pytorch and ESRGAN. 

## About
The Super-Resolution Generative Adversarial Network (SRGAN) is a seminal work that is capable of generating realistic textures during single image super-resolution. However, the hallucinated details are often accompanied with unpleasant artifacts. To further enhance the visual quality, we thoroughly study three key components of SRGAN - network architecture, adversarial loss and perceptual loss, and improve each of them to derive an Enhanced SRGAN (ESRGAN). In particular, we introduce the Residual-in-Residual Dense Block (RRDB) without batch normalization as the basic network building unit. Moreover, we borrow the idea from relativistic GAN to let the discriminator predict relative realness instead of the absolute value. Finally, we improve the perceptual loss by using the features before activation, which could provide stronger supervision for brightness consistency and texture recovery. Benefiting from these improvements, the proposed ESRGAN achieves consistently better visual quality with more realistic and natural textures than SRGAN and won the first place in the PIRM2018-SR Challenge.

**Credit:** 
ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks

Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Chen Change Loy, Yu Qiao, Xiaoou Tang

[Paper](https://arxiv.org/pdf/1609.04802) 

[Code(PyTorch)](https://github.com/xinntao/ESRGAN)

## Preview

![Screenshot (130)](https://user-images.githubusercontent.com/101701760/172054099-dcf043fe-c208-4f39-900a-8256392dcdc8.png)


## Requirements To Run on Google Colab:
To run a demo or carry out testing with the API it's best to do that with Google Colab. To run/test the API on Google Colab do the following:
1. Clone the repository.
2. Open a Google Colab instance and upload the **.ipynb** file and the **requirements_colab.txt** to that instance.
3. Run each cell until the last cell and you should be able to view the API with a link that has the name **ngrok** in it.

## Requirements to Run on Local Machine
1. Clone the repository to your local machine
1. Install the requirements from the requirements.txt file:
```
pip install -r requirements.txt
```
2. Then from your command line run:
```
python -m uvicorn --port 5000 --host 127.0.0.1 app:app --reload 
```
Then you can view the site on your local server: http://127.0.0.1:5000/ 

## Building Docker image
To build the docker container image;
1. Install docker
1. Clone the repository
2. Then from your command line run:
```
docker build . -t [image-name]

```

## Deployment
The api can be deployed using the dockerfile or the procfile on heroku.

### Deployment with Dockerfile
Assuming you have git and heroku cli installed just carry out the following steps:

1. Clone the repository

```
git clone https://github.com/Nneji123/Super-Resolution-Model.git
```

2. Change the working directory

```
cd Super-Resolution-Model 
```

4. Create the heroku app

``` 
heroku create your-app-name 
```

Replace **your-app-name** with the name of your choosing.

4. Set the heroku cli git remote to that app

```
heroku git:remote your-app-name
```

5. Set the heroku stack setting to container
 
```
heroku stack:set container
```

6. Push to heroku
```
git push heroku main
```

### Deployment with Github
1. Clone the repository and copy the contents to your own repository on github
2. Deploy with Github and Heroku then you're done.


## Live Link
The api and its documentation can be viewed here: https://super-resolution-api.herokuapp.com/docs or https://super-resolution-api.herokuapp.com/redoc




## Live Link
[Image Super Resolution API](https://super-resolution-api.herokuapp.com/docs)



