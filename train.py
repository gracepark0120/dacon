import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import kfp

def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

class SatelliteDataset(Dataset):
    def __init__(self, csv_file, transform=None, infer=False):
        self.data = pd.read_csv(csv_file) # 데이터 로드
        self.transform = transform
        self.infer = infer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1] #self.data의 idx번째 행의 1번 열에 위치한 이미지 경로
        if './train_img' in img_path:
          new_path = img_path.replace('./train_img', './app/train_img')
        elif './test_img' in img_path:
          new_path = img_path.replace('./test_img', './app/test_img')

     #   image = cv2.imread(new_path)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 이미지는 OpenCV를 사용하여 로드하고 RGB로 변환.
        if self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
            return image # 추론 모드 시 이미지만 반환

        mask_rle = self.data.iloc[idx, 2] # self.data의 idx번째 행의 2번 열에 위치한 RLE(Run-Length Encoding)로 인코딩된 마스크 정보를 가져옵니다.
        mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))
        # 훈련 모드 시 rle 문자열로 인코딩된 마스크를 디코딩, 변환하여 이미지와 마스크를 반환.
        # RLE로 인코딩된 마스크 정보를 디코딩하여 원래 형태의 마스크로 변환합니다. 변환된 마스크의 크기는 이미지의 높이와 너비와 동일합니다.
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask
    
train_transform = A.Compose(
        [
            A.OneOf(
                [
                    A.RandomBrightness(p=1),
                    A.RandomBrightnessContrast(p=1),
                    A.Emboss(p=1),
                    A.RandomShadow(p=1),
                    A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, fill_value=0, p=0.5),
                    A.NoOp(),

                ],
                p=1,
            ),
            A.OneOf(
                [
                    A.Blur(p=1),
                    A.AdvancedBlur(p=1),
                    A.MotionBlur(p=1),
                ],
                p=0.6,
            ),
            A.OneOf(
                [
                    A.NoOp(),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.ShiftScaleRotate(p=0.5),
                    A.Rotate(limit=90, p=1, border_mode=cv2.BORDER_REPLICATE),
                    A.RandomRotate90(p=1)
                ],
                p=1,
            ),
            A.ElasticTransform(),
            A.RandomCrop(224, 224),
            A.Normalize(),
            ToTensorV2(transpose_mask=True)
        ]
    )


test_transform = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2()
    ]
)

train_dataset = SatelliteDataset(csv_file='./app/train.csv', transform=train_transform)
test_dataset = SatelliteDataset('./app/test.csv', transform=test_transform, infer=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=8)


import torch
import argparse
from tqdm import tqdm
import numpy as np
# 모델 초기화
# GPU를 사용할 경우


logs = []
# loss function과 optimizer 정의
#criterion = HybridLoss()
#criterion = DiceLoss()
criterion = torch.nn.BCEWithLogitsLoss() # 이진 교차 엔트로피
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

torch.autograd.set_detect_anomaly(True)

for epoch in range(3):  # 10 에폭 동안 학습합니다.
    model.train()
    epoch_loss = 0

    for images, masks in tqdm(train_dataloader):
        images = images.float().to(device)
        masks = masks.float().to(device)

        optimizer.zero_grad()
        out = model(images)

        loss = criterion(out, masks.unsqueeze(1))

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(train_dataloader)}')

    log_epoch = {'epoch': epoch+1,'train_loss': epoch_loss}
    logs.append(log_epoch)
    scheduler.step(epoch_loss)  # 학습률 스케쥴러 호출



torch.save(model.state_dict(), 'practice1_' + str(epoch+1) + '.pth')

