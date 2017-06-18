# Data

<!-- TOC depthFrom:2 -->

- [Example Datasets](#example-datasets)

<!-- /TOC -->

## Example Datasets

The MNIST datasets (http://yann.lecun.com/exdb/mnist/) were re-formatted into comma-separated text files, with one sample per line, for 10-class and binary (0 vs 1) classification problems. Move the training and testing files to appropriate locations when you build and run the clients and the server.

These are included in the attached `.zip` files. Unpack them through your local *unzip* command (e.g. on unix `unzip <file>`) and the resulting file structures should be produced below:

- `mnist-regular/`
    - `MNISTTrainImages.dat` - 10,000 samples of 784-dimension data
    - `MNISTTrainLabels.dat` - 10,000 samples of 10-class labels from 0-9
    - `MNISTTestImages.dat`  - 1,000 samples of 784-dimension data
    - `MNISTTestLabels.dat`  - 1,000 samples of 10-class labels from 0-9
- `binary-mnist/`
    - `binaryTrainImages.dat` - 12,445 samples of 785-dimension data
    - `binaryTrainLabels.dat` - 12,445 samples of 2-class labels from 0-1
    - `binaryTestImages.dat`  - 220 samples of 785-dimension data
    - `binaryTestLabels.dat`  - 220 samples of 2-class labels from 0-1
