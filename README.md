# temporalnetv2modified
Modified script for temporalnetv2, not for end user.

**How to use:**
1.Following original guide from https://huggingface.co/CiaraRowles/TemporalNet2
2.Install everything you need, instead of running the default one in that link, use the python script here.


The original script have problem in file grabbing as it would grab 1.png and 10.png instead of 1.png and 2.png as optical flow method frame interpolation should, this is due to not natural sorting the images.
The script allow use of text2img with hiresfix enabled, with hiresfix enabled, 24G vram is suggested, without hiresfix, it can run in 12G vram.
The original script does not allow preprocessed perfect mask, nor do it allow user to input different image for different cnet, in the modified script you should modify the path to each cnet image folder with the variable, thus allowing workflow for automatic mask output via blender.
This script act as a prove of concept of my workflow that would be opensauced later, please try it and share result with me. I am also trying to make a comparsion between txt2img and img2img, so far img2img hold the original color better while txt2img allow more style changes.
