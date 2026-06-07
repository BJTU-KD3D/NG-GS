# NG-GS: NeRF-Guided 3D Gaussian Splatting Segmentation

NG-GS is a framework for high-quality object segmentation in 3D Gaussian Splatting (3DGS). It addresses the boundary discretization artifacts that commonly occur at semantic edges by introducing a **NeRF-guided boundary continuity** stage on top of boundary-adaptive Gaussian splitting.

The framework operates in two core stages:

- **Edge Gaussian Continuity**: Ambiguous Gaussians located at object boundaries are identified through mask variance analysis. These boundary Gaussians are used to build a spatially continuous feature field via RBF interpolation, and a multi-resolution hash encoding (MRHE) enhances the representation capacity while keeping the computation efficient.
- **NeRF-GS Joint Optimization**: The interpolated and hash-encoded features are fed into a lightweight NeRF module that acts as a continuous refinement network. A joint optimization strategy with alignment, continuity, and smoothness losses harmonizes the outputs of 3DGS and NeRF, so the final segmentation keeps high-frequency boundary details with smooth transitions across views.

## Environment Setup
To prepare the environment, 

1. Clone this repository. 
	```
	git clone https://github.com/your-org/NG-GS.git
	```
2. Follow [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) to install dependencies. 
   	```
	conda env create --file environment.yml
    conda activate NG-GS
	```
	Please notice, that the ```diff-gaussian-rasterization``` module contained in this repository has integrated the mask training branch to implement ```Boundary-Adaptive Gaussian Splitting```.

3. Install [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2).
   
   We provide a stable sequence masks extraction method based on Grounded-SAM-2 in ```./submodules/Grounded-SAM-2-utils```.
	```
	cd submodules
    git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
    cd Grounded-SAM-2 
    cd checkpoints
    wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
    cd ..
    cd gdino_checkpoints
    wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
    cd ..

    pip install -e .
    pip install --no-build-isolation -e grounding_dino

    cd ../..
    cp ./submodules/Grounded-SAM-2-utils/grounded_sam2_tracking_demo.py ./submodules/Grounded-SAM-2
	```
    

## Run NG-GS

We provide ```process.sh``` to easily implement the complete segmentation process, which only requires the image sequence of the scene and the text prompts of the segmented parts.

1. Train 3DGS
  ```
    python train.py -s "dataset/tandt/truck" -m "output/truck"  --images "images_4"
  ```
2. Extract masks based on text prompt
  ```
    python submodules/Grounded-SAM-2/grounded_sam2_stable_tracking.py --dataset "tnt" --output "output" --scene "truck" --text "The truck" --resolution 4
  ```
3. Run NG-GS segmentation
   
  ```
    python train.py -s "dataset/tandt/truck" -m "output/truck" --start_checkpoint "output/truck/chkpnt30000.pth" --include_mask --finetune_mask --text "The truck" --images "images_4" --N4views 14 --mask_signals_threshold 0.8 --use_ngs --ngs_max_points 4096
  ```
  - ```--include_mask```: Add mask to the render.
  - ```--finetune_mask```: Split the boundary Gaussian using mask gradient. Using only ```include_mask``` does not change the structure of the scene.
  - ```--N4views```: ```L``` images, additionally optimize ```L*N4views``` epochs.
  - ```--mask_signals_threshold```: Threshold of relative distance. 
  - ```--use_ngs```: Enable the NeRF-guided boundary continuity stage (Edge Gaussian Continuity + NeRF-GS joint optimization).
  - ```--ngs_max_points```: Maximum number of boundary Gaussians sampled per step for NeRF-guided refinement.
  - ```--lambda_ngs_align``` / ```--lambda_ngs_cont``` / ```--lambda_ngs_smooth```: Weights of the alignment, continuity, and smoothness losses, respectively.

Noting the need for fair comparison, we provide [masks](https://drive.google.com/drive/folders/1mMwj1510hb0PMEnxjUpzIDe2N3EL2PUF?usp=sharing) obtained on the [NVOS dataset](https://jason718.github.io/nvos/) based on points prompts. Under our project, just put them under the ```./output``` folder and skip ```Extract masks based on text prompt```. Finally different scenes are evaluated in ```eval/eval_NVOS.py```

### TODO List
- [ ]  Update efficient multi-object segmentation.
- [ ]  Update efficient texture optimizations.
- [ ]  Provide demo and more visualizations.

## Citation
*If you find this project helpful for your research, please consider citing the report and giving a ⭐.*

*Any questions are welcome for discussion.*
```
@inproceedings{he2026ng,
  title={NG-GS: NeRF-Guided 3D Gaussian Splatting Segmentation},
  author={He, Yi and Wang, Tao and Jin, Yi and Lang, Congyan and Li, Yidong and Ling, Haibin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={42061--42070},
  year={2026}
}
```
