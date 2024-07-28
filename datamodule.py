from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from datasets_flow import VSSLFlowDataset
from torchvision import transforms
from PIL import Image
import os
import csv


def convert_to_vggss(filename):
    # prefix = filename[0:11]
    # suffix = int(filename[12:].replace(".mp4", ""))
    # return f"{prefix}_{suffix*100}_{(suffix+10)*1000}"
    return filename.replace(".mp4", "")


class NoPriorDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()

        self.args = args

    def setup(self, stage):
        self.train_dataset = self.setup_train_dataset(self.args)
        self.val_dataset = self.setup_val_dataset(self.args)

    def train_dataloader(self):
        dl = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.args.batch_size,
            pin_memory=False,
            drop_last=True,
            shuffle=True,
            num_workers=self.args.num_workers
        )

        return dl

    def val_dataloader(self):
        dl = DataLoader(
            dataset=self.val_dataset,
            batch_size=1,
            pin_memory=False,
            drop_last=False,
            num_workers=self.args.num_workers
        )

        return dl

    def test_dataloader(self):
        dl = DataLoader(
            dataset=self.val_dataset,
            batch_size=1,
            pin_memory=False,
            drop_last=False,
            num_workers=self.args.num_workers
        )

        return dl

    def setup_train_dataset(self, args):
        audio_path = os.path.join(args.train_data_path, "audio/")
        img_path = os.path.join(args.train_data_path, "frames/")
        flow_path = os.path.join(args.train_data_path, "flow/")

        dataset = open(f"metadata/{args.trainset}.txt").read().splitlines()

        img_items = os.listdir(img_path)
        img_items = [item.replace(".jpg", "") for item in img_items]
        for i in range(len(dataset)):
            if dataset[i] not in img_items:
                print(f"Removing {dataset[i]} from dataset")
                dataset[i] = None

        dataset = [item for item in dataset if item is not None]
        print(len(dataset))

        if 'vgg' in args.trainset:
            dataset = [convert_to_vggss(item) for item in dataset]

        audio = []
        frames = []

        for sample in dataset:
            audio.append(sample + ".wav")
            frames.append(sample + ".jpg")

        audio = sorted(audio)
        frames = sorted(frames)

        audio_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0], std=[12.0])])

        print("Train Dataset Initialized")

        return VSSLFlowDataset(
            mode='train',
            img_files=frames,
            audio_files=audio,
            img_path=img_path,
            flow_path=flow_path,
            audio_path=audio_path,
            concat_num=args.concat_num,
            audio_transform=audio_transform
        )

    def setup_val_dataset(self, args):
        audio_path = os.path.join(args.test_data_path, "audio/")
        img_path = os.path.join(args.test_data_path, "frames/")
        flow_path = os.path.join(args.test_data_path, "flow/")

        if args.testset == 'flickr':
            testcsv = 'metadata/flickr_test_trimmed.csv'
        elif args.testset == 'flickr_expanded':
            testcsv = 'metadata/flickr_test_expanded.csv'
        elif args.testset == 'vggss':
            testcsv = 'metadata/vggss_test_trimmed.csv'
        elif args.testset == 'music_duet':
            testcsv = 'metadata/music_duet_test.csv'
        elif args.testset == 'music_solo':
            testcsv = 'metadata/music_solo_test.csv'

        data = []
        with open(testcsv) as f:
            csv_reader = csv.reader(f)
            for item in csv_reader:
                data.append(item[0])

        img_items = os.listdir(img_path)
        img_items = [item.replace(".jpg", "") for item in img_items]
        for i in range(len(data)):
            if data[i] not in img_items:
                print(f"Removing {data[i]} from dataset")
                data[i] = None
        flow_x_items = os.listdir(os.path.join(flow_path, "flow_x/"))
        flow_x_items = [item.replace(".jpg", "") for item in flow_x_items]
        for i in range(len(data)):
            if data[i] is not None and data[i] not in flow_x_items:
                print(f"Removing {data[i]} from dataset")
                data[i] = None
        flow_y_items = os.listdir(os.path.join(flow_path, "flow_y/"))
        flow_y_items = [item.replace(".jpg", "") for item in flow_y_items]
        for i in range(len(data)):
            if data[i] is not None and data[i] not in flow_y_items:
                print(f"Removing {data[i]} from dataset")
                data[i] = None

        data = [item for item in data if item is not None]

        # Modify VGGSS strings if using VGGSS
        if 'vggss' in args.testset:
            data = [convert_to_vggss(item) for item in data]

        audio = []
        frames = []

        for sample in data:
            audio.append(sample + ".wav")
            frames.append(sample + ".jpg")

        audio = sorted(audio)
        frames = sorted(frames)

        audio_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0], std=[12.0])])

        print("Test Dataset Initialized")

        return VSSLFlowDataset(
            mode='test',
            img_files=frames,
            audio_files=audio,
            img_path=img_path,
            flow_path=flow_path,
            audio_path=audio_path,
            concat_num=args.concat_num,
            audio_transform=audio_transform,
        )
