
# dense_lstm_denoise

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Getting Started

### Prerequisites

- **[Anaconda](https://docs.anaconda.com/anaconda/install/):** Package and environment manager.
- **Python >= 3.8**
- **Git** (optional, for cloning the repository)

### Installation

1. **Clone the Repository**

   ```sh
   git clone https://github.com/forXuyx/dense_lstm_denoise.git
   cd dense_lstm_denoise
   ```

2. **Download the dataset or generate data by yourself**

   - **Download the dataset:**

   10,000 samples, download from [Baidu Netdisk](https://pan.baidu.com/s/1MsyKh5X934Ybldr4U9SNRQ) (code: 9t9j).

   - **Generate data by yourself:**

   you need to setup the lisa environment, these packages can download from [Baidu Netdisk](https://pan.baidu.com/s/1SArmfCfwOKYGI6gHgYJwJQ) (code: feyq).

   once you download the packages, you should run the following command to setup the lisa environment:

   ```sh
   unzip lisa_all.zip
   cd lisa_all
   bash install.sh
   ```

   those commands will create a new conda environment named `lisa_all`, then you just need to activate the environment and run the data_generation.ipynb to generate the data.


3. **Train the model(Pytorch needed)**

   ```sh
   python train.py --config configs/config.json 
   ```

