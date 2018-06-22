​	

### 	                                          Semi-Supervised Hand Pose Estimation with Deep Convolutional GAN
___


Environment:

- Ubuntu16.04 LTS
- TensorFlow　1.4
- Python　2.7
- OpenCV　3.2.0


The train data is augmented NYU dataset from [deep-prior-pp](https://github.com/moberweger/deep-prior-pp). We can get 152750 depth images, each image shape is (128, 128)

​	Accurate estimation of the 3D pose in real-time has many challenges, including the presence of local self-similarity and self-occlusions. We propose to model the statistical relationships of 3D hand poses and corresponding depth images. By design, our architecture allows for learning from unlabeled image data in a semi-supervised manner. 

![](http://ww1.sinaimg.cn/large/006zLtEmgy1fsgcj3yt27j31v70xyjun.jpg)

Overview of the proposed system, Joints stand for hand pose Conv_T stands for transposed convolutional layers with dialation factor of 2, and Conv stands for convolutional layers with stride of 2, and FC denotes fully convolutional layers. One neural network, called the generator, generates the new depth hand  images, while the other, called the discriminator, evaluates them for authentically, decides whether each intance of depth hand images it reviews belongs to the actual training dataset or not. To prevent from overfitting, we formulate the hand pose estimation as mulitask learning in which all tasks share the first and second convolutional layers.

​	A GAＮ consists of  a generator and a discriminator. The generator synthesizes samples by  mapping from joints distribution to a sample in data space $x$ . The discriminator tries to distinguish between real data samples $x$ and synthesized samples $\hat x$ from generator. The loss function for the GAN can be formulated as a binary entropy loss as follows:                               

![](http://latex.codecogs.com/gif.latex?%24%24%7B%7B%5Ctext%7BL%7D%7D_%7Bgan%7D%7D%3D%5Clog%20%28Dis%28x%29%29&plus;%5Clog%20%281-Dis%28Gen%28J%29%29%29%20%24%24)

In order to shorten training time and obtain more real synthesized images, we introduce a  proxy loss $L_{recons}​$, based  on error between the synthesized images and real images, which can lead the network to converge faster and search better local minimum. Similar to the golden energy, we use a clipped mean squared error for our loss function, to remain robust to depth sensor noise. The loss function can be formulated as follows: 

![](http://latex.codecogs.com/gif.latex?%24%24%5Ctext%7BL%7D_%7Brecons%7D%3D%5Cfrac%7B1%7D%7BN%7D%5Csum%5Climits_%7Bi%7D%5E%7BN%7D%7B%5Cmax%28%7C%7C%7Bx%5E%7B%28i%29%7D%7D-Gen%28J%5E%7B%28i%29%7D%29%7C%7C%5E2%2C%5Ctau%20%29%7D%20%24%24)

   Given an input depth images $X$，to obtain the hand pose estimation, meanwhile, we can encourage the discriminator to more accurately distinguish the fake images, we formulate a loss function as follows:

![](http://latex.codecogs.com/gif.latex?%24%24%7B%7B%5Ctext%7BL%7D%7D_%7BJ%7D%7D%3D%5Cfrac%7B1%7D%7BN%7D%5Csum%5Climits_%7Bi%7D%5E%7BN%7D%7B%7C%7CDis%28%7B%7BX%7D%5E%7B%28i%29%7D%7D%29-%28J%29%7C%7B%7B%7C%7D%5E%7B2%7D%7D%7D%24%24)

​	In each iteration, both the generator and the discriminator are jointly updated. The discriminator is updated with labeled, unlabeled and synthesized samples, at the same time, the generator is updated through back-propagated gradients. The joint update ensures that the generator synthesizes more realistic samples for the discriminator. So we can define the joint generator and discriminator loss as:

![](http://latex.codecogs.com/gif.latex?%24%24%7B%5Ctext%7BL%7D%7D_%7BGen%7D%3DL_%7Brecons%7D-%7BL%7D_%7Bgan%7D%24%24)

![](http://latex.codecogs.com/gif.latex?%24%24%7B%7B%5Ctext%7BL%7D%7D_%7BDis%7D%7D%3D%7B%7BL%7D_%7BJ%7D%7D&plus;%7B%7BL%7D_%7B%5Ctext%7Bgan%7D%7D%7D%24%24)

where $L_{Gen}$ represents the generator loss and $L_{Dis}$ the discriminator loss.

The raw depth images as follows:

|                             NO.1                             |                            NO.２                             |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="http://ww1.sinaimg.cn/large/006zLtEmgy1fsginlfsynj30hs0dcglw.jpg" width="400"/> | <img src="http://ww1.sinaimg.cn/large/006zLtEmgy1fsginlh1mxj30hs0dcmxg.jpg" width="400"/> |

The synthesized images from the generator as follows:

|                             NO.1                             |                            NO.２                             |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="http://ww1.sinaimg.cn/large/006zLtEmgy1fsgioanhehj30hs0dcmxy.jpg" width="400"/> | <img src="http://ww1.sinaimg.cn/large/006zLtEmgy1fsgioan6lgj30hs0dcgmd.jpg" width="400"/> |

We set the learning rate as 0.001 and train the complete network for 50 epochs. It takes about 3.42 hours for training with 140k samples on one Nvidia GTX 1080TI.

![](http://ww1.sinaimg.cn/large/006zLtEmgy1fsgin02p0mj30m80gowes.jpg)

