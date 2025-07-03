# CLIP-steroids: CLIP on Steroids
Train CLIP family models with high flexibility of customization.
Unleash Swarm of Zero Shot Models

<p align="center">
  <img src="https://github.com/anish9/CLIP-steroids/blob/main/assets/clipster.png" alt="ClipSteroids Logo" width="600"/>
</p>

### About
CLIP-steroids is a flexible, high-performance training framework that lets you mix and match any image encoder with any text decoder to build optimized CLIP-style models.

🚀 Why CLIP-steroids?
  - Combine any image encoder with any text decoder — full flexibility.
  - Out-of-the-box support for **25+ image backbones** and **10+ text backbones.**
  - Easily configurable with a ```config.yaml``` file — no hard coding required.
  - Built for rapid experimentation — iterate fast, succeed faster.
  - Optimize for performance or model size as needed.

  Just set up your config and launch training. That’s it.

### Training set-up 
```
pip install -r requirements.txt
```
- A sample dataset format is provided in the ```dataset``` folder.
- Edit the **config.yaml** file.
  - A model catalog file is present at ```clipkit/model_catalog.py```.
  - We can couple any **Image backbone** with **Text backbone** to get our optimal model based on our Dataset *complexity*.
    
  ```
  Training:
    batch_size: 8
    learning_rate: 5e-5
    epochs: 10
    ckpt_dir: "model_ckpt"  #model checkpoints dir.
    logs_dir: "model_logs"  #tensorflow model logs.
    ckpt_max_keep: 3        #maximum files to store in the checkpoint dir(overwrites).

  Model:
    image_encoder: "EfficientNetB1" #select any from clipkit/model_catalog.
    tune_image_layers_count: 10
    text_encoder: "google/electra-small-discriminator" #select any from clipkit/model_catalog.
    tune_text_layers: True
    proj_dim: 512
    max_length: 12 #max text sequence length.
  
  Dataset:
    train_csv_path: "dataset/demodata.csv"
    val_csv_path: "dataset/test.csv"

  ```
  That's it!
- Run
  ```
  python train.py
  ```
