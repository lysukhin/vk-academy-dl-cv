```shell
python tools/create-segmentation-dataset.py PATH_TO_EXTRACTED_DATA
python tools/train-segmentation.py -d PATH_TO_EXTRACTED_DATA -o runs/segmentation_baseline

python tools/create-recognition-dataset.py PATH_TO_EXTRACTED_DATA
python tools/train-recognition.py -d PATH_TO_EXTRACTED_DATA -o runs/recognition_baseline

python tools/create-submission.py -d PATH_TO_EXTRACTED_DATA --seg_model SEG_CKPT --rec_model REG_CKPT -o SUBMISSION_NAME
# or use (pretrained) segmentation_baseline.pth & recognition_baseline.pth
```

* [Competition page](TODO)
* [Pretrained checkpoints (baseline)](https://disk.yandex.ru/d/O1s4217MMbpwHw)
