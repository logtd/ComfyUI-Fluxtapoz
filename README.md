# ComfyUI-Fluxtapoz

A set of nodes for editing images using Flux in ComfyUI

## Examples

See `example_workflows` directory for examples.

No ControlNets are used in any of the following examples.

## Rectified Flow Inversion (Unsampling from [RF Inversion](https://rf-inversion.github.io/))

Admittedly this has some small differences between the example images in the paper, but it's very close. Will be updating as I find the issue.
It's currently my recommended way to unsample an image for editing or style transfer.

Use [this workflow](https://github.com/logtd/ComfyUI-Fluxtapoz/blob/main/example_workflows/example_rf_inversion_updated.json) for RF-Inversion.

![rf_inversion](https://github.com/user-attachments/assets/f0517649-4dbb-4371-a8d5-3ae90e3b6368)

### RF-Inversion Stylization

RF-Inversion can also be used to stylize images.

Use [this workflow](https://github.com/logtd/ComfyUI-Fluxtapoz/blob/main/example_workflows/example_rf_inversion_stylization.json) to style images.

![rf_inversion_style](https://github.com/user-attachments/assets/bb1a1aef-74f1-4ad3-9687-d3d8d41ce7d1)


## RF-Edit (Unsampling from [RF-Solver-Edit](https://github.com/wangjiangshan0725/RF-Solver-Edit))
RF-Edit is an alternative way to edit images. It may suit some use cases better than RF-Inversion and I recommend trying both.

Use [this workflow](https://github.com/logtd/ComfyUI-Fluxtapoz/blob/main/example_workflows/example_rf_edit_workflow_alternative.json) for RF-Edit.

![rf_edit_wolf](https://github.com/user-attachments/assets/ad583ad0-7e04-4bc5-9d3c-f08d4cc493a0)

## Fireflow (Unsampling from [Fireflow inversion](https://github.com/HolmesShuan/FireFlow-Fast-Inversion-of-Rectified-Flow-for-Image-Semantic-Editing))
For a faster inversion method there is also Fireflow for image editing. 

Use [this workflow](https://github.com/logtd/ComfyUI-Fluxtapoz/blob/main/example_workflows/example_rf_fireflow.json) for Fireflow.

![fireflow_pirate_example](https://github.com/user-attachments/assets/0e8f8da5-6340-4d42-8d84-afedf0afe753)

## Flow Edit
This is an implementation of image editing from [FlowEdit](https://github.com/fallenshock/FlowEdit).

It is an inversion free way to edit images.

Use [this workflow](https://github.com/logtd/ComfyUI-Fluxtapoz/blob/main/example_workflows/example_flow_edit.json) to get started.

![flux_flowedit](https://github.com/user-attachments/assets/a1aaeb8f-5eeb-448c-a206-b678b097fd3c)



## Regional Prompting
Regional prompting allows you to prompt specific areas of the latent to give more control. You can combine it with Redux, but Redux is so powerful it dominates the generation.
This implementation is based on [InstantX Regional Prompting](https://github.com/instantX-research/Regional-Prompting-FLUX).

Use [this workflow](https://github.com/logtd/ComfyUI-Fluxtapoz/blob/main/example_workflows/example_flux_regional.json) for regional prompting.

![ComfyUI_temp_kzsib_00029_](https://github.com/user-attachments/assets/364ea890-8585-43f4-9ffa-f0a24a775f27)


## Enhancement
There are two nodes for Perturbed Attention Guidance (PAG) and Smoothed Energy Guidance (SEG) that can add detail to images.

The following from left to right: Vanilla Flux, PAG, SEG
![ComfyUI_00069_](https://github.com/user-attachments/assets/6e8bf072-dcc8-454e-bf97-b76a5694db91)



## Acknowledgements
Thank you to all researchers involved in the tools implemented in this repo.

<details>
<summary>Click to see all acknowledgements</summary>
  
[RF-Inversion](https://rf-inversion.github.io/)

```
@article{rout2024rfinversion,
  title={Semantic Image Inversion and Editing using Rectified Stochastic Differential Equations},
  author={Litu Rout and Yujia Chen and Nataniel Ruiz and Constantine Caramanis and Sanjay Shakkottai and Wen-Sheng Chu},
  journal={arXiv preprint arXiv:2410.10792},
  year={2024}
}
```

[RF-Solver-Edit](https://github.com/wangjiangshan0725/RF-Solver-Edit)
```
@article{wang2024taming,
  title={Taming Rectified Flow for Inversion and Editing},
  author={Wang, Jiangshan and Pu, Junfu and Qi, Zhongang and Guo, Jiayi and Ma, Yue and Huang, Nisha and Chen, Yuxin and Li, Xiu and Shan, Ying},
  journal={arXiv preprint arXiv:2411.04746},
  year={2024}
}
```

[Fireflow](https://github.com/HolmesShuan/FireFlow-Fast-Inversion-of-Rectified-Flow-for-Image-Semantic-Editing)
```
@misc{deng2024fireflowfastinversionrectified,
      title={FireFlow: Fast Inversion of Rectified Flow for Image Semantic Editing}, 
      author={Yingying Deng and Xiangyu He and Changwang Mei and Peisong Wang and Fan Tang},
      year={2024},
      eprint={2412.07517},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.07517}, 
}
```

[FlowEdit](https://github.com/fallenshock/FlowEdit)
```
@article{kulikov2024flowedit,
	title = {FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models},
	author = {Kulikov, Vladimir and Kleiner, Matan and Huberman-Spiegelglas, Inbar and Michaeli, Tomer},
	journal = {arXiv preprint arXiv:2412.08629},
	year = {2024}
	}
```

[InstantX Regional Prompting](https://github.com/instantX-research/Regional-Prompting-FLUX)
```
@article{chen2024training,
  title={Training-free Regional Prompting for Diffusion Transformers},
  author={Chen, Anthony and Xu, Jianjin and Zheng, Wenzhao and Dai, Gaole and Wang, Yida and Zhang, Renrui and Wang, Haofan and Zhang, Shanghang},
  journal={arXiv preprint arXiv:2411.02395},
  year={2024}
}
```
</details>
