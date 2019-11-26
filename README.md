# Detect StyleGAN generated images using CNNs (for Miyazaki-san) 

## Requirements
```
pip install -r requirements.txt
```

## Training
Start visdom first：

```
python -m visdom.server
```

Then start training via:

```
# train on gpu0, with no existing checkpoints
python main.py train --load-model-path=None --use-gpu=True
```


For detail instruction, use:
```
python main.py help
```

## Testing

```
python main.py test --data-root=./your_test_dataset --use-gpu=False --batch-size=256
```
