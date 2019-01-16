import torch.utils.data as data
import random
from PIL import Image
import os
<<<<<<< HEAD
import numpy as np 
import cv2
=======
import numpy as np
import cv2

>>>>>>> 3c4d900bd566e94cfac6788d25cdcfbbe88a0996

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class TSNDataSet(data.Dataset):
<<<<<<< HEAD
    def __init__(self, root_path,list_file,num_segments=3,new_length=1,
                 modality='RGB',image_tmpl='{:05d}.jpg',transform=None,
                 random_shift=True,test_mode=False,On_Video=False
=======
    def __init__(self, root_path, list_file, num_segments=3, new_length=1,
                 modality='RGB', image_tmpl='{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False, On_Video=False,interval=1
>>>>>>> 3c4d900bd566e94cfac6788d25cdcfbbe88a0996
                 ):
        super(TSNDataSet, self).__init__()
        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
<<<<<<< HEAD
        self.On_Video=On_Video
=======
        self.On_Video = On_Video
        self.interval=interval
>>>>>>> 3c4d900bd566e94cfac6788d25cdcfbbe88a0996

        if self.modality == 'RGBDiff':
            self.new_length += 1

        self._parse_list()

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):

        record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) \
                if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_val_indices(record)

        return self._get(record, segment_indices)

    def _sample_indices(self, record):
        average_duration = (record.num_frames - self.new_length * self.interval + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + \
                      np.random.randint(0, average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments if self.modality == 'RGB' else self.new_length * self.interval:
            offsets = np.sort(random.randint(record.num_segments - self.new_length * self.interval + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length * self.interval - 1:
            tick = (record.num_frames - self.new_length * self.interval + 1) / float(self.num_segments)
            offsets = np.array([int(tick * x) for x in range(self.num_segments)])
        else:
<<<<<<< HEAD
            offsets=np.zeros((self.num_segments,))
        return offsets+1

    def _get(self,record,indices):
        images=list()

        if self.On_Video==False:
            for seg_ind in indices:
                p=int(seg_ind)
                for i in range(self.new_length):
                    seg_imgs=self._load_image(os.path.join(self.root_path,record.path),p)
                    images.extend(seg_imgs)
                    if p<record.num_frames:
                        p+=1
        else:
            cap=cv2.VideoCapture(os.path.join(self.root_path,record.path))
            for seg_ind in indices:
                p=int(seg_ind)
                for i in range(self.new_length):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, p)
                    _, frame = cap.read()
                    seg_imgs=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) 
                    images.extend(seg_imgs)
                    if p<record.num_frames:
                        p+=1
            cap.release()

        #(segment*sample_len,w,h,c)
        process_data=self.transform(images)
        return process_data,record.label

    def _load_image(self,path,idx):
=======
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get(self, record, indices):
        images = list()

        if self.On_Video == False:
            for seg_ind in indices:
                p = int(seg_ind)
                for i in range(self.new_length):
                    seg_imgs = self._load_image(record.path, p)
                    images.extend(seg_imgs)
                    if p < record.num_frames:
                        p += self.interval
        else:
            cap = cv2.VideoCapture(os.path.join(self.root_path, record.path + '.avi'))
            for seg_ind in indices:
                p = int(seg_ind) - 1
                for i in range(self.new_length):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, p)
                    res, frame = cap.read()
                    seg_imgs = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    images.append(seg_imgs)
                    if p < record.num_frames:
                        p += self.interval
            cap.release()

        # (segment*sample_len,w,h,c)
        process_data = self.transform(images)
        return process_data, record.label

    def _load_image(self, record_path, idx):
>>>>>>> 3c4d900bd566e94cfac6788d25cdcfbbe88a0996
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            path = os.path.join(self.root_path, record_path)
            return [Image.open(os.path.join(path, self.image_tmpl.format(idx))).convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(self.root_path, self.image_tmpl.format('u', record_path, idx))).convert('L')
            y_img = Image.open(os.path.join(self.root_path, self.image_tmpl.format('v', record_path, idx))).convert('L')
            return [x_img, y_img]

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split()) for x in open(self.list_file)]


if __name__ == '__main__':
    '''
    train_loader=data.DataLoader(
        TSNDataSet("","../raw/test_list.txt",num_segments=3,
            new_length=1,modality="RGB",
            image_tmpl="{:05d}.jpg" if args.modality in ['RGB','RGBDiff'] else args.flow_prefix+'{}_{:05d}.jpg',
            tranform=torchvision.tranforms.Compose([
                train_augmentation,
                Stack(roll=args.arch=='BNInception'),
                ToTorchFormatTensor(div=args.arch!='BNInception'),
                normalize
                ])
            ),
        batch_size=args.batch_size,shuffle=True,
        num_workers=args.workers,pin_memory=True
        )
    '''
    pass
