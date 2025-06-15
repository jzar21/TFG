# Development of a Deep Learning-Based System for Age Estimation Using Radiological Knee Images

<div>
  <a href="https://github.com/jzar21/TFG/blob/main/LICENSE">
    <img alt="Code License" src="https://img.shields.io/github/license/jzar21/TFG"/>
  </a>

  <img src="https://img.shields.io/pypi/pyversions/torch"/>
</div>

## Description

This repository contains the code and resources related to my Bachelor’s Thesis at the University of Granada. The main goal of the project is to develop a deep learning model capable of estimating an individual's age from radiological images of the knee.

## Requirements

To run this project, the following dependencies are required:

* Python 3.8 or higher
* PyTorch
* scikit-learn
* numpy
* matplotlib
* pandas
* opencv-python

These dependencies are listed in the `requirements.txt` file.

## Installation

To set up the development environment, follow these steps:

1. Clone this repository:

   ```bash
   git clone https://github.com/jzar21/TFG.git
   cd TFG
   ```

2. Create and activate a virtual environment (recommended using `conda`):

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To train the model, run the following command:

```bash
python main.py config.json
```

## Contributions

Contributions to the project are welcome. If you’d like to collaborate, please follow these steps:

1. Fork this repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and ensure that all tests pass.
4. Submit a pull request detailing your changes.

## License

This project is licensed under the GNU General Public License v3.0. For more details, see the `LICENSE` file.
