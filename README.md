## Usage

 - Install OpenPose. Installation instructions :
   https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md
 
 - Use face_detect.py to create a 96x96 sized cropped image of the face in the image.  Change the working directory to the OpenPose directory downloaded in the first instruction and execute the following command to get a keypoint-heatmap of the face. The output will be stored in the all_heatmaps folder of the same OpenPose directory.

> ./build/examples/openpose/openpose.bin --video location/of/image --
heatmaps_add_parts --heatmaps_add_bkg --heatmaps_add_PAFs --model_pose COCO
--display 0 --render_pose 0 --net_resolution 96x96 --write_heatmaps output_
heatmaps_folder/

 
 - Add the location of the newly generated heatmap to the 'own_set' list in main.py and execute the code.
 
 - The angles are normalized, so the resultant tensor with the yaw,pitch and roll should be multiplied by 180 and then 90 be subtracted from every element to get theese angles in degrees.


## Credits -
 - Research paper by - Aryaman Gupta, Kalpit Thakkar, Vineet Gandhi and P J Narayanan
   https://arxiv.org/pdf/1812.00739.pdf

 
