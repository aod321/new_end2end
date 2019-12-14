# new_end2end
## Return to the original idea

##Model Structure
![WX20191214-201439@2x.png](https://i.loli.net/2019/12/14/JAkVyKWraOCgx9N.png)
## A rough introduction
This work is extended from [1], trying to use a STN(or affine_transform)
to link the separate Stage1(PartsCropper Stage) and Stage2(Parts Segmentation Stage), 
so that they can finnally be trained End2End. Hopefully it should get a better result
since usually a end2end joint training will brings improvements for both stage(like the Faster R-CNN, Mask R-CNN).

## Latest Result
Still need to imporve
![WX20191214-202410@2x.png](https://i.loli.net/2019/12/14/hBFOJVIUCoWvAaT.png)

## Reference
[1]Zhou, Yisu, Xiaolin Hu, and Bo Zhang. "Interlinked convolutional neural networks for face parsing." International symposium on neural networks. Springer, Cham, 2015.
APA	
