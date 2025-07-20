# Task2 Generate Images using Pre-Trained models
Here is a *detailed and long-form README.md* for your project using *Stable Diffusion with Diffusers* and Google Colab, themed around *AI + Nature Fusion*. This includes setup instructions, usage steps, saving images, and collaboration info.


## 🌱🧠 Stable Diffusion Image Generation – AI + Nature Fusion Theme(can also any other prompt)

This project uses the *Stable Diffusion v1.4 model* from the diffusers library by Hugging Face to generate AI art with the theme *"AI + Nature Fusion". It is built and tested on **Google Colab* and is ideal for artists, developers, students, and AI enthusiasts who want to explore generative AI for visual creativity.


## 📌 Project Objective

The goal of this project is to create visually stunning, AI-generated images that fuse natural elements (trees, flowers, landscapes, wildlife) with technological or futuristic motifs (robots, cyborgs, machinery) using *prompt-based generation*.



## 🧠 Tools & Technologies

* 🧠 *HuggingFace Diffusers*
* 🖼 *Stable Diffusion v1.4*
* 🔧 *PyTorch*
* 📘 *Transformers*
* 🌐 *Google Colab* (Cloud-based Jupyter Notebook)
* 💾 *Pillow (PIL)* for image saving
* 📦 *Jupyter nbconvert* (optional cleanup)



## 📂 File Structure

.
├── stable_diffusion.ipynb         # Main Colab Notebook
└── README.md                      # Project documentation


## 🚀 Setup Instructions

### 📌 Prerequisites

* Google account (for Colab)
* Basic understanding of Python
* Optional: Git + GitHub for version control

### ▶ Run on Google Colab

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload your stable_diffusion.ipynb file
3. Run the notebook step-by-step

Or click this badge to launch directly:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)



## 🛠 Installation (if using locally)

If you want to run this notebook locally via Jupyter, follow these steps:

pip install torch torchvision torchaudio
pip install diffusers transformers accelerate scipy safetensors
pip install notebook


Then launch:

jupyter notebook




## ✨ How It Works

1. *Load Pre-trained Model* from HuggingFace Hub:


from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16
).to("cuda")


2. *Set a Prompt* related to AI + Nature:


prompt = "a futuristic robot meditating under a cherry blossom tree"


3. *Generate and Save Image:*


image = pipe(prompt).images[0]
image.save("fusion_art.png")


4. *Optional: Add Seed for Reproducibility*


generator = torch.Generator("cuda").manual_seed(1024)
image = pipe(prompt, generator=generator).images[0]




## 📸 Prompt Ideas 

 To guide our prompts and image generation, pick a theme. Example options:

* Fantasy Creatures
* Futuristic Cities
* AI + Nature Fusion
* Surreal Dreams
* Book Cover Concepts
* Mythology Reimagined
* Fashion Designs
  

## 💾 Saving Images to Local System

To save generated images:


image.save("ai_nature_fusion_output.png")


To download them from Colab:


from google.colab import files
files.download("ai_nature_fusion_output.png")



## 📄 License

This project is for educational and personal use. Commercial use of Stable Diffusion may require additional licensing. See [CreativeML License](https://github.com/CompVis/stable-diffusion/blob/main/LICENSE) for details.



## 🤝 Credits

* HuggingFace 🤗 diffusers team
* Stability AI for Stable Diffusion
* Google Colab for GPU support
* You! For exploring the intersection of *AI + Nature*
