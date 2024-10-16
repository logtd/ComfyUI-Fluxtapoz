# ComfyUI-Fluxtapoz (WIP)

A set of nodes for editing images using Flux in ComfyUI

## Examples

See `example_workflows` directory for examples.

No ControlNets are used in any of the following examples.

## Rectified Flow Inversion (Unsampling from [RF Inversion](https://rf-inversion.github.io/))
Admittedly this has some small differences between the example images in the paper, but it's very close. Will be updating as I find the issue.
It's currently my recommended way to unsample an image for editing or style transfer.
![rf_inversion](https://github.com/user-attachments/assets/f0517649-4dbb-4371-a8d5-3ae90e3b6368)

It can also be used to mix or style images (although I'm still working out the settings for this)
![rf_inverse_mix](https://github.com/user-attachments/assets/2588fab7-3de6-4708-b1da-6da4c8be4edb)



### Node Parameters

#### Outverse Flux Model Pred Node
* Ensure "reverse_ode" is set to True on the "Outverse Flux Model Pred" node. Sometimes when users upgrade this repo it doesn't load the workflow correctly.

#### Flux Reverse ODE Sampler
* latent_image -- the image to guide the sampling
* start_step -- the step that the sampler starts guiding the sampling towards the image in "latent_image"
* end_step -- the last step for guiding the sampling (not inclusive)
* eta -- the strength of the guidance

#### Flux Forward ODE Sampler
* gamma -- the paper leaves this at 0.5, but is similar to eta in the Reverse sampler

#### Guidance Suggestions
* For sampling normal flux guidance works (~3.5)
* For unsampling I find low values such as 0 or 1 work best

### Inverse Noise (unsampling)

![unsampling_example](https://github.com/user-attachments/assets/9c604a31-5cc9-49c2-9a08-98e7872591c2)

### Inject Inversed Noise

See example workflow for how to use this one. It's similar to inverse noise/unsampling, but has better adherence to the input image.

![inject_inversed_noise_example](https://github.com/user-attachments/assets/ee052855-12c6-47f7-8178-b4acfb2ca6b9)
![inject_unsampled_noise_cowboy](https://github.com/user-attachments/assets/4d92c591-e04d-4123-a432-d859a32e5f46)


### Reference Sampling (WIP)
This is a different implementation than TDS' and is experimental. Some results can turn out blurry and it is difficult to control.
See the updated workflow json for the latest test.

![ComfyUI_temp_tpfdf_00010_](https://github.com/user-attachments/assets/2233ae6a-e26d-4ca7-9493-b4cdefe2b470)

![wolf_ref](https://github.com/user-attachments/assets/739fe969-ed9b-43ae-aea2-586075bf2288)

![anime_ref_example](https://github.com/user-attachments/assets/4245bf2c-db2a-494b-adec-cc1f30ffda0b)

