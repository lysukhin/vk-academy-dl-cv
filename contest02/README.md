```
pip install editdistance

PYTHONPATH=. python scripts/create-segmentation-dataset.py --data-dir PATH_TO_EXTRACTED_DATA
PYTHONPATH=. python scripts/train-segmentation.py -d PATH_TO_EXTRACTED_DATA -o runs/segmentation_baseline
# or use pretrained/segmentation_10ep.pth

PYTHONPATH=. python scripts/create-recognition-dataset.py --data-dir PATH_TO_EXTRACTED_DATA
PYTHONPATH=. python scripts/train-recognition.py -d PATH_TO_EXTRACTED_DATA -o runs/recognition_baseline
# or use pretrained/recognition_24ep.pth

PYTHONPATH=. python scripts/create-submission.py -d PATH_TO_EXTRACTED_DATA --seg_model SEG_CKPT --rec_model REG_CKPT -o SUBMISSION_NAME
```

* [Pretrained checkpoints](https://cloud.mail.ru/public/L2xs/f51o45up4)
* [Competition page](https://www.kaggle.com/c/made-cv-2021-contest-02-license-plate-recognition/host/sandbox-submissions)
