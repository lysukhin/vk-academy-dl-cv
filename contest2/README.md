This pipeline consists of detection through segmentation + recognition.
Python 3.7.6, Pytorch 1.4.0

For segmentation/recognition datasets you've got to run create_datasets.ipynb to convert bboxes to segmentation masks and crop them for OCR.

Each task can be trained using train.py in the respective dir. All the parameters, used to obtain the baseline,
 are set to default, except data_path.

Once the 2 models are trained (you can find the baseline models in pretrained sections), you can proceed to inference.py and generate a submission. 

Current networks and routines are basic, there's plenty of room for improvement and quick wins.
 The "TODO" sections are scattered across the code and hint to the possible problems/avenues of research, but TODOs are 
 not extensive, so do not rely solely on them.

The pipeline is by no means optimal, you can extend it by introducing new components (neural networks/heuristics),
 especially between the two current networks.
 
Also the baseline overfits and doesn't score good on the private part of the test dataset, which comes from a bit different distribution. 

Good luck and have fun!


PS. Some unusual requirements:
 - pip install editdistance
