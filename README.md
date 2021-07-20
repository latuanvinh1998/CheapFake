# CheapFake
## Testing on MMSys 2021

### For training, data training and validation folder must be in ../Data/train/ and ../Data/val

### For testing, folder must also in ../Data/public_test_mmsys/, annotations file in ..Data/mmsys_anns/

#### Here we proposal three method, Cosine Similarity, Euclidean Distance and Classifier method.


##### For Training, please run:
```bash
python train_xxx.py
```


##### For Evaluate, please run:
```bash
python eval_xxx.py
```

Replace xxx with: cosine for Cosine Similarity Method, triplet for Euclidean Distance method and classify for Classifier method.
