# LECSFormer

[Refined Crack Detection Via LECSFormer for Autonomous Road Inspection Vehicles](https://doi.org/10.1109/TIV.2022.3204583)

**Abstract:**

Due to the rising cost of human resources in road maintenance and the pursuit of efficiency, autonomous road inspection vehicles are developed for intelligent detection of road disease to prevent severe traffic disasters in the early stages. Nevertheless, as a prevalent road disease, road cracks are diverse and susceptible to shadows, weather changes, and noise in data acquisition. Moreover, they usually appear with thin shapes that are hard to detect correctly by existing methods. To handle this problem, more details of the road cracks need to be better analyzed. In this article, we propose a refined road crack detection method named locally enhanced cross-shaped windows transformer (LECSFormer), which adopts a delicate design of the encoder-decoder structure. The encoder employs window-based transformer blocks to model long-range dependencies. Each transformer block ensembles the locally enhanced module to enrich the local contextual information, and token shuffle operation is applied to build cross-windows connections. The decoder uses dense connections to fuse multi-scale information, and the feature fusion module fuses hierarchical features and reweights them by the channel attention mechanism. The proposed method outperforms other state-of-the-art methods with ODS of 0.963, 0.917, 0.952, and 0.953 on four challenging datasets, CrackTree260, CrackLS315, Stone331, and CRKWH100. It can accurately detect cracks in road surfaces and support intelligent preventive maintenance of roads.
<!-- ![Crack detection for autonomous road inspection vehicles](https://github.com/ZhaoNan1/LECSFormer/blob/main/images/fig1.png) -->
<p align="center">
  <img src="https://github.com/ZhaoNan1/LECSFormer/blob/main/images/fig1.png"> <br>
  Crack detection for autonomous road inspection vehicles
</p>

<p >
  <img src="https://github.com/ZhaoNan1/LECSFormer/blob/main/images/LECSFormer.png"> <br>
  The proposed Locally Enhanced Cross-Shaped Windows Transformer Architecture consisted of the encoder, decoder, and feature fusion model (FFM),where the LECSFormer blocks are composed of a LECSFormer block and a Shuffle LECSFormer block.
</p>

# Results

## CrackTree260
| Model | ODS | OIS | AP | #param. |
|:------|:----|:----|:----|:----|
| Ours | 0.963 | 0.969 | 0.969 | 16.34M |

## CrackLS315
| Model | ODS | OIS | AP | #param. |
|:------|:----|:----|:----|:----|
| Ours | 0.917 | 0.927 | 0.915 | 16.34M |

## Stone331
| Model | ODS | OIS | AP | #param. |
|:------|:----|:----|:----|:----|
| Ours | 0.952 | 0.970 | 0.960 | 16.34M |
# Get started

## Training
`python train.py --root_path path/datasets/  --cfg configs/config.yaml`
## Testing

`python test.py --root_path path/datasets/  --cfg configs/config.yaml  --output_dir output/`

# Citation
If our code or models help your work, please cite our papers:
```BibTeX
@ARTICLE{9878251,
  author={Chen, Junzhou and Zhao, Nan and Zhang, Ronghui and Chen, Long and Huang, Kai and Qiu, Zhijun},
  journal={IEEE Transactions on Intelligent Vehicles}, 
  title={Refined Crack Detection Via LECSFormer for Autonomous Road Inspection Vehicles}, 
  year={2022},
  volume={},
  number={},
  pages={1-13},
  doi={10.1109/TIV.2022.3204583}}
```

# Acknowledgement
[Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)

[CSWin-Transformer](https://github.com/microsoft/CSWin-Transformer)

[Shuffle Transformer](https://github.com/mulinmeng/Shuffle-Transformer)

Thanks for the great implementations!
