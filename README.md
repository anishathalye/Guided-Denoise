# Guided-Denoise

The code in this repository demonstrates that [Defense against Adversarial
Attacks Using High-Level Representation Guided
Denoiser](https://arxiv.org/abs/1712.02976) (Liao et al. 2018) is ineffective
in the white-box threat model.

With an L-infinity perturbation of 4/255, we generate targeted adversarial
examples with 100% success rate.

See our note for more context and details: https://arxiv.org/abs/1804.03286

## Pretty pictures

Obligatory picture of sample of adversarial examples against this defense.

![](hgd.jpg)

## Citation

```
@unpublished{cvpr2018breaks,
  author = {Anish Athalye and Nicholas Carlini},
  title = {On the Robustness of the CVPR 2018 White-Box Adversarial Example Defenses},
  year = {2018},
  url = {https://arxiv.org/abs/1804.03286},
}
```

## [robustml] evaluation

Run with:

```bash
cd nips_deploy
python robustml_attack.py --imagenet-path <path>
````

[robustml]: https://github.com/robust-ml/robustml

### Credits

Thanks to [Dimitris Tsipras](https://github.com/dtsip) for writing the robustml
model wrapper.
