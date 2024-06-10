import h5py
import torch
import numpy as np

device = "cuda:1"

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

        with h5py.File(self.images, mode='r') as f:
            self.file_names = list(f.keys())

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        with h5py.File(self.images, mode='r') as i:
            image = i[file_name][:]
        with h5py.File(self.masks, mode='r') as m:
            mask = m[file_name][:]
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image)
        mask = mask.astype(np.uint8)

        return image, mask

def compute_dsc(prediction, ground_truth, class_label):
    intersection = np.sum((prediction == class_label) & (ground_truth == class_label))
    union = np.sum((prediction == class_label) | (ground_truth == class_label))

    dsc = (2.0 * intersection) / (union + intersection)
    return dsc * 100  # Convert to percentage

def computeDSC(dataloader, model):
    model.eval()
    size = len(dataloader.dataset)
    test_loss = 0

    dsc = np.zeros(17, dtype=np.float32)

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            pred = model(x).to("cpu")
            pred = torch.argmax(pred, dim=1)
            pred = pred.numpy()
            pred = np.squeeze(pred)
            y = y.numpy()

            dscsum = 0
            for cl in range(1, 17):
                class_dsc = compute_dsc(pred, y, cl)
                dsc[cl] += class_dsc
                dscsum += class_dsc
            
            dsc[0] += (dscsum / 16)
                
    dsc = dsc / size

    return dsc


test_data = CustomDataset("WORD_Val_img.h5", "WORD_Val_mask.h5")
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
model = torch.load("trained_3DUnetDiceCE.pth")
model.to(device)
dicescores = computeDSC(test_loader, model)


clNumtoOrgan = {0: "average",
                1: "liver",
                2: "spleen",
                3: "left_kidney",
                4: "right_kidney",
                5: "stomach",
                6: "gallbladder",
                7: "esophagus",
                8: "pancreas",
                9: "duodenum",
                10: "colon",
                11: "intestine",
                12: "adrenal",
                13: "rectum",
                14: "bladder",
                15: "Head_of_femur_L",
                16: "Head_of_femur_R"}



for cl in range(17):
    print(f"{clNumtoOrgan[cl]}: {dicescores[cl]:0.2f}")