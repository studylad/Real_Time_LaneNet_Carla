# Real Time Lane Detection in Carla with LaneNet (v1.0)

This provides lane line segmentation from LaneNet in the autonomous driving research simulator Carla. Below is a video showing its functionality.


[![lanenet carla](https://i.imgur.com/Y7q1tBf.png)](https://www.youtube.com/watch?v=47nwbZrG-aU "lanenet carla")


The LaneNet implementation is done in TensorFlow and is from @MaybeShewill-CV. The repository is [here](https://github.com/MaybeShewill-CV/lanenet-lane-detection).

The original paper of LaneNet: [Towards End-to-End Lane Detection: an Instance Segmentation Approach](https://arxiv.org/abs/1802.05591)

For the data pipeline from Carla, code from [carla-simulator/data-collector](https://github.com/carla-simulator/data-collector) was modified.

## Installation

1. Following instructions and install [MaybeShewill-CV/lanenet-lane-detection](https://github.com/MaybeShewill-CV/lanenet-lane-detection)

2. Following instructions and install [carla-simulator/data-collector](https://github.com/carla-simulator/data-collector) as well as CarlaGear (i.e. Carla 0.8.4) (see carla-simulator/data-collector readme)

3. Merge respective files from the repos above into the files in this repo.

4. Download the folder containing the frozen LaneNet graph [here](https://drive.google.com/drive/folders/1zdJHBHgF_MSC7XXCy7zr9vqmU25rww2z?usp=sharing) and place the folder into the lanenet_lane_detection folder.

## Usage

1. Launch CarlaGear session from the console:

```bash
sh CarlaUE4.sh /Game/Maps/Town01 -windowed -world-port=2000 -benchmark -fps=30 -quality-level=Low
```

2. In separate console window from the repo's directory, run:
```bash
 python run_carla_lanenet_session.py 
```

## Future Work

- [ ] Get points working instead of imposing mask onto image.
- [ ] Achieve higher frame rate.
- [ ] Create agent to steer car between detected lanes.
- [ ] Create Carla lane dataset and train LaneNet on it.

## Contributing
I would be pleased to collaborate with others!

## Citations

@MaybeShewill-CV https://github.com/MaybeShewill-CV/lanenet-lane-detection

@carla-simulator https://github.com/carla-simulator/data-collector
