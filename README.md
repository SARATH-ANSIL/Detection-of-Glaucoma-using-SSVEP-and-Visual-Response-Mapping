# Detection of Glaucoma using SSVEP and Visual Response Mapping

This project implements a machine learning model for detecting glaucoma by analyzing Steady-State Visual Evoked Potentials (SSVEP) from EEG data. The model leverages deep learning techniques to provide accurate classification based on visual responses.

## Features
- **Multi-task Learning:** Uses a multi-task learning approach to improve classification accuracy for different glaucoma severity levels.
- **EEG Signal Analysis:** Processes EEG signals in response to visual stimuli for disease detection.
- **Model Visualization:** Provides visualizations of input and output tensors for better understanding of the model's performance.

## Prerequisites
- Python 3.x
- The following Python libraries must be installed:
  - `torch`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `mne`
  - `pyriemann`
  - `scikit-learn`
  - `pandas`
  
You can install the required libraries using pip:
```bash
pip install torch numpy matplotlib seaborn mne pyriemann scikit-learn pandas
```
## Project Structure

```bash
.
├── models/                # Contains model architecture and training scripts
│   ├── multitask_model.py # Script defining the multi-task learning model
│   ├── train.py           # Script for training the model
│   └── ...                # Other necessary model files and dependencies
├── output/                # Folder to save output images and results
├── detection.ipynb        # Main Jupyter Notebook for model training and testing
└── README.md              # Project README file
```
## How It Works
- **Model Architecture**: The model is built using PyTorch, implementing convolutional layers to extract features from the EEG signals.
- **Data Preparation**: The dataset is prepared and formatted to match the model's expected input shape.
- **Model Training**: The multi-task learning model is trained on the EEG data to classify different glaucoma conditions.
- **Model Evaluation**: The model is evaluated using validation data, and results are visualized using heatmaps for better understanding.
- **Inference**: The trained model can be used for inference on new EEG data to detect glaucoma.
- 
## Running the Project
Open the Jupyter Notebook:
`jupyter notebook detection.ipynb`

Follow the instructions in the notebook to load your dataset and initialize the model.

Prepare your dataset and ensure it is correctly formatted.

Run the training script to train the model on your EEG dataset.

Test the model using validation data to see the detection results.

Visualize the output using heatmaps for analysis.

## Example Output
After running the training and testing scripts, the results will be saved in the output folder, showing the heatmaps of the EEG signals and the corresponding detection results for glaucoma.

## Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page for new ideas.

## License
This project is licensed under the MIT License.

Acknowledgments
PyTorch for deep learning capabilities.
MNE for processing EEG data.
NumPy and Matplotlib for data handling and visualization.
