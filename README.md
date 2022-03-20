Assignment 3
===================================

<!DOCTYPE html>
<html>
<img src="https://raw.githubusercontent.com/learning3d/learning3d.github.io/main/data/late_days/two.png"   width=200px height=200px>
</html>

Rohan Agarwal (rohanaga)

Late days: 2

## Question 1
```
Command: python main.py --config-name=box
```
### 1.3. Ray sampling Visualizations

![](images/out/1_3/image_grid.png) ![](images/out/1_3/ray_grid.png)


###  1.4. Point sampling Visualization

![](images/out/1_4/sample_pt_cloud.png)

### 1.5. Volume rendering Visualization
![](images/part_1.gif) ![](images/out/1_5/depth_map.png)

##  Question 2. Optimizing a basic implicit volume
```
Command: python main.py --config-name=train_box
```
```
Box center: (0.2502, 0.2506, -0.0005)
Box side lengths: (2.0051, 1.5036, 1.5034)
```

![](images/part_2.gif)

##  Question 3. Optimizing a Neural Radiance Field (NeRF) 
```
Command: python main.py --config-name=nerf_lego_no_view
```
Rendering WITH positional encoding and WITHOUT view dependence:

![](images/part_3_noview.gif)


##  Question 4. NeRF Extras

###  4.1 View Dependence Visualization
```
Command: python main.py --config-name=nerf_lego
```

Rendering WITH positional encoding and WITH view dependence:

![](images/part_3.gif)


**Comments:**

1. With view-dependency added fine-grained detail and lighting effects are rendered much better.
For instance, the shadows on the base of the bulldozer and the shine on the blade of the bulldozer 
when facing the light source are more apparent.
2. Spurious flicker in the rendering is also reduced with view-dependency. 

3. Viewpoint conditioning is added after density estimation i.e. much later in the network because:
    a) it only affects the color component and 
    b) exposing the whole network to view encoding may lead to over fitting.

4. NeRF allows for arbitrary lighting and is rather too flexible. Without enforcing a physically-informed lighting model, 
NeRF can tend to have low generalization power i.e. it can perform poorly when cameras are imprecise and viewpoints
are sparse. However, given dense set of precise views adding viewpoint dependence helps improve rendering quality.


###  4.3 High Resolution Imagery Visualization

```
Command: python main.py --config-name=nerf_lego_med_highres
Command: python main.py --config-name=nerf_lego_highres
```

| ![](images/part_4_med_highres.gif) | ![](images/part_4.gif) | 
|:----------------------------------:|:----------------------:|
|      *low-capacity network*   |  *high-capacity network*    |


**Comments:**

In general: 

1. Higher fine-grained details ( for instance the lego base or the grey grill-like structure 
on the front of the bulldozer and the shaft joints) are visible with high-res rendering. The effect of viewpoint conditioning 
is also much more apparent on the lego base and on the blade of the bulldozer where you can see shadows. 
2. However, it takes a significant increase in the network capacity (depth, number of hidden dimensions, position and 
 view dependence encoding to achieve a higher representational power. This means, while NeRF allows for rendering arbitrarily 
 resolutions, it comes at the cost of high computation power.


I experimented with two configuration:

High resolution (400 x 400) with low network capacity:
```
  batch_size: 1024
  image_size: [400, 400]
  chunk_size: 32768
  n_harmonic_functions_xyz: 8
  n_harmonic_functions_dir: 3
  n_hidden_neurons_xyz: 128  
  n_hidden_neurons_dir: 64
  n_layers_xyz: 7
  append_xyz: [3]
```
High resolution (400 x 400) with high network capacity:
```
  batch_size: 512
  chunk_size: 8192
  n_harmonic_functions_xyz: 10
  n_harmonic_functions_dir: 4
  n_hidden_neurons_xyz: 256
  n_hidden_neurons_dir: 128
  n_layers_xyz: 8
  append_xyz: [4]
 ```
  
As expected, the rendering with a higher network capacity has a sharper rendering. The smaller network also results
in more spurious rendering which reflects as flickering in the rendering (can be seen on a closer look). 
    
But what I find worth-noting is the fact that the network parameters had to increase more than 4 times in number 
for what doesn't seem like a lot of improvement visually. This again points towards how the computation requirement 
for training can scale significantly even for small improvements in rendering quality.
