# SALT - Simply Annotate by Leveraging Tracking

A lightweight utility for fast bounding box and segmentation mask annotations. It is fast because it is built on a visual tracking backbone to minimize manual input and is especially lightweight since it leverages the simple but powerful opencv graphical toolkit. 

SALT was born out of frustration with [CVAT](https://github.com/openvinotoolkit/cvat) when working with egocentric videos (eg, from a GoPro). In these videos, the effects of camera and object motion necessitate frequent user input and drag down annotation times. Additionally, the server client model of CVAT is extremely slow to buffer videos. We leverage the excellent tracking of [SiamMask](http://www.robots.ox.ac.uk/~qwang/SiamMask) to work around these bottlenecks while trading-off for a sophisticated UI. Of course, this is only useful with a GPU machine and the setup is complicated.

The idea for the project came from another related project [Anonymal](https://github.com/ezelikman/anonymal).

Features include:

1. Playback/rewind/stepping through the video
2. Multiple object tracking
3. Terminate/delete tracks
4. Pre-populate annotations 

## Demo
<div align="center">
  <img src="demo.gif" />
</div>
<br>

Watch the [screencast](https://youtu.be/9Gp0ORBF66o) on youtube.

## Installation
Follow the setup instructions from SiamMask's [repo](https://github.com/foolwood/SiamMask#environment-setup). Don't forget to change the clone url to this repo's url and env vars/paths as needed.

Tested on Ubuntu 18.04, Python 3.7, Pytorch 1.5.0, and CUDA 11.1.

## Usage

```shell
cd <PROJECT_ROOT>/experiments/siammask_sharp
export PYTHONPATH=$PWD:$PYTHONPATH
python ../../tools/video_cleaner.py --base_path foo.mp4
```
SALT has four primary modes of usage: playback, review, create_bbox & modify. In each mode, the available commands are displayed as text near the top of the video. The descriptions below focus on the purpose behind each and switching between modes.

1. Playback - As the name suggests, this mode is for video playback. Playback is fast and beats the need for a media player. In addition, any objects annotated in the past and yet active are tracked as well, labeling frames as they are played. Standard play/pause and skip ahead options available using keyboard.

2. Review - When users pause the video in playback mode, they enter review mode which allows stepping through frames at a granular level, allowing accurate pin-pointing for object start/end times. From the review mode, three transitions are possible:
    - playback - Start playing again.
    - create_bbox - Annotate a new object and start tracking/labeling it in future video frames. Draw the bounding box and hit Space/Enter when ready.
    - modify - Terminate an existing object track (eg: if the object has gone out of frame) or delete it altogether. A track terminated in the past (inactive track) is displayed with a red border and can only be deleted. If there are multiple tracks, the tool cycles through them displaying one at a time.
  
__Note__: For each new track, a deep learning model is initialized. Therefore, the speed and memory usage is proportional to number of _active_ tracks. Terminating a track deletes the model and frees up memory.

At the end, mask annotations are saved in the parent directory of the video file in a folder with same name as the video minus extension.

## Bibtex
If you find this code useful, please consider citing 
```
@misc{salt2021,
    title={SALT - Simply Annotate by Leveraging Tracking},
    author={Sharma, Jayant},
    booktitle={GitHub},
    year={2021}
}
```

In addition, the implementation and method are based closely on the SiamMask paper by Wang et. al (2019), so please consider citing:

```
@inproceedings{wang2019fast,
    title={Fast online object tracking and segmentation: A unifying approach},
    author={Wang, Qiang and Zhang, Li and Bertinetto, Luca and Hu, Weiming and Torr, Philip HS},
    booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
    year={2019}
}
```

## License
Licensed under an MIT license.

