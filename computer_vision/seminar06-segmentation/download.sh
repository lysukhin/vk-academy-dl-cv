export URL='https://raw.githubusercontent.com/lysukhin/MADE/2022/computer_vision/seminar06-segmentation/'
wget "$URL/augmentations.py"
wget "$URL/seminar.py"
mkdir resources
wget "$URL/resources/example-image.jpg" -O resources/example-image.jpg
wget "$URL/resources/unet-window.jpg" -O resources/unet-window.jpg
wget "$URL/resources/00-1-noaug.jpg" -O resources/00-1-noaug.jpg
wget "$URL/resources/02-pretrained.jpg" -O resources/02-pretrained.jpg
wget "$URL/resources/01-1-aug.jpg" -O resources/01-1-aug.jpg
wget "$URL/resources/00-2-noaug.jpg" -O resources/00-2-noaug.jpg
wget "$URL/resources/made.jpg" -O resources/made.jpg
wget "$URL/resources/example-mask.jpg" -O resources/example-mask.jpg
wget "$URL/resources/00-3-noaug.jpg" -O resources/00-3-noaug.jpg
wget "$URL/resources/u-net.jpg" -O resources/u-net.jpg
mkdir cvmade
mkdir cvmade/plot
wget "$URL/cvmade" -O cvmade
wget "$URL/cvmade/plot/plot.py" -O cvmade/plot/plot.py
wget "$URL/cvmade/plot/__init__.py" -O cvmade/plot/__init__.py
wget "$URL/cvmade/plot/torch.py" -O cvmade/plot/torch.py
wget "$URL/cvmade/__init__.py" -O cvmade/__init__.py
wget "$URL/cvmade/torch.py" -O cvmade/torch.py
wget "$URL/cvmade/image.py" -O cvmade/image.py

