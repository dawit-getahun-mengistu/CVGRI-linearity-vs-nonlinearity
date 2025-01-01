training commands:

```
# Train AlexNet with default parameters
python train.py --model alexnet

# Train linear DeepConvNet with custom parameters
python train.py --model lnrdeepconv --batch_size 64 --learning_rate 0.001 --patience 7

# Train with different number of classes
python train.py --model alexnet --num_classes 4 --epochs 30
```

After training visualization can be done with the following commands


vizualization commands:

```
# Using features before final layer
python viz.py --model alexnet --checkpoint model_checkpoints/alexnet_best.pth --use_features

python viz.py --model lnrdeepconv --checkpoint model_checkpoints/lnrdeepconv_best.pth

# Custom save directory and perplexity
python viz.py --model alexnet --checkpoint model_checkpoints/alexnet_best.pth \
    --save_dir my_plots --perplexity 40 --use_features
```
