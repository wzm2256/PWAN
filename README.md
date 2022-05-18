# Partial Wasserstein Adversarial Network for Non-rigid Point Set Registration

This is a Pytorch implement of the PWAN model proposed in [Partial Wasserstein Adversarial Network for Non-rigid Point Set Registration](https://openreview.net/forum?id=2ggNjUisGyr).


Partial Wasserstein Adversarial Network (PWAN) seeks to partially match two unnormalized distributions. It can be seen as a generalization of the popular Wasserstein GAN model to the unbalanced case. The code in this repository applies PWAN to point set registration problems.

|Point sets| Potential| Gradient| 
|--------------|--------------|--------------|
<img src="Readme_fig\fish_vis.png" width="256"/>  | <img src="Readme_fig\Fish_m3.png" width="256"/> |<img src="Readme_fig\Fish_m3_grad.png" width="256"/>

## Requirements
- pytorch=1.4
- open3d (for visualizing 3D point sets)
- matplotlib (for visualizing 2D point sets)
- pykeops (for fine-tuning registration)


## Usage
Common usage is shown in scrip.py, and see more detailed explanations of the parameters in main.py.

## Reference


    @inproceedings{wang2022partial,
        title={Partial Wasserstein Adversarial Network for Non-rigid Point Set Registration},
        author={Zi-Ming Wang and Nan Xue and Ling Lei and Gui-Song Xia},
        booktitle={International Conference on Learning Representations (ICLR)},
        year={2022}
    }

For any question, please contact me (wzm2256@gmail.com).