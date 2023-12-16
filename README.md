# FallNLP2023

Create a new conda environment (we use Python version 3.10), and install the following packages:
```
conda install datasets transformers accelerate pytorch-base==1.12.1
```

To use the HuggingFace API to generate the C4 dataset with a given number of datapoints (we use 100000):
```
python generate_dataset.py -n 100000
```
This generates a json file containing the dataset.

To train the denoising model with a given number of datapoints (n) and a batch size (b) (we use 16):
```
python driver.py -n 100000 -b 16
```
The model weights will be saved under a folder ex. `./checkpoint-2000` (checkpoint after 2000 training steps)

To run evaluation using the downstream task pipeline, specify the path to the desired denoising model checkpoint:
```
python eval.py --path ./checkpoint-2000
```

To obtain baseline results from a pretrained summarization model on the CNN dataset:
```
python cnn_baseline.py
```
