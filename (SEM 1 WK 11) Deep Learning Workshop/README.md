# Introduction to Deep Learning

Hello and welcome to the Deep Learning workshop! This workshop covers fundamental concepts in Convolutional Neural Networks (CNNs) and Transformer architectures with hands-on examples using PyTorch.

## Requirements

### Environment Setup
This workshop is designed to work best with **Google Colab** (free, web-based IDE). While you can run the notebooks locally in VSCode or Jupyter, we **strongly recommend using Google Colab** for:
- **GPU acceleration** (essential for the IMDB Transformer training)
- **Interactive features** (text input prompts work seamlessly)
- **Pre-installed libraries** (no local setup headaches)

To use Google Colab:
1. Go to [Google Colab](https://colab.research.google.com/)
2. Upload the notebooks from this repository
3. Change Runtime Type to GPU: `Runtime` → `Change runtime type` → `Hardware accelerator: GPU`

### Local Setup (Alternative)
If you prefer a local IDE (PyCharm/VSCode), ensure you have:
- Python 3.8 or higher installed
- An Integrated Development Environment (IDE) configured
- GPU support (recommended for Transformer training)

You can install the required libraries by running:

**For CNN notebooks (CNN_Playground, Visualisation_of_Feature_Maps_and_Kernels):**
```bash
pip install -r CNN/requirements.txt
```

**For Transformer notebook:**
```bash
pip install -r Transformer/requirements.txt
```

**Note:** The interactive text input features in the Transformer notebook may not work as smoothly in VSCode compared to Google Colab.

## Dataset

The workshop uses the following datasets:

- **`IMDB Dataset.csv`**: 50,000 movie reviews for sentiment classification (located in `Transformer/` folder)
- **`cat.png`**: Sample image for CNN visualization examples (located in `CNN/` folder)

Please ensure these files are in the correct locations before running the notebooks. If using Colab, upload the `IMDB Dataset.csv` to the same directory as the notebook.

## Code

The workshop code is provided in Jupyter notebook (`.ipynb`) files that are designed to run interactively in Google Colab or other notebook environments. The notebooks contain:
- **Markdown cells**: Explanations, theory, and instructions
- **Code blocks**: Runnable Python code with inline comments

### Notebooks Overview:

1. **`CNN/CNN_Playground.ipynb`**: Hands-on experimentation with CNN architectures on MNIST dataset
2. **`CNN/Visualisation_of_Feature_Maps_and_Kernels.ipynb`**: Visualize how CNNs "see" images through feature maps
3. **`Transformer/Transformers.ipynb`**: 
   - Text summarization using Transformer architecture
   - Mini sentiment classifier (10 reviews)
   - Full IMDB sentiment classification (50K reviews) - **Requires GPU**

### Running the Notebooks:

**In Google Colab (Recommended):**
1. Upload the notebook to Colab
2. Change runtime to GPU for Transformer training
3. Run cells sequentially from top to bottom
4. Interactive features (like text input) will work seamlessly

**In VSCode/Jupyter:**
1. Install extensions for Jupyter Notebooks support
2. Open the `.ipynb` file
3. Run cells sequentially
4. Note: Interactive input prompts may require terminal-based execution

Should you have any questions during the workshop, feel free to raise them! Happy coding!



## Additional Resources

Helpful resources to familiarize yourself with GitHub and version control:
- [Intro to Github](https://docs.github.com/en/get-started/start-your-journey/hello-world)
- [Walkthrough Github Desktop](https://docs.github.com/en/desktop/overview/getting-started-with-github-desktop)
- [Git Gud for beginners](https://www.youtube.com/watch?v=8Dd7KRpKeaE)
- [Must Know Git Commands](https://github.blog/developer-skills/github/top-12-git-commands-every-developer-must-know/)
- [Github in VSCode](https://code.visualstudio.com/docs/sourcecontrol/github)
