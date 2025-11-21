print("Importing...")
import babble_data
import numpy as np
import time
from PIL import Image
from tqdm import tqdm
import cv2
import torch.nn as nn

import multiprocessing
multiprocessing.freeze_support() 
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

from trainer_distsampler import read_capture_file

from models import MicroChad, MultiInputMergedMicroChad
import torch.nn.functional as F
import torch
import random
import sys
import random
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

try:
    import torch_directml
    device = torch_directml.device()
except:
    device = "cpu"

print("Preparing dataset...")
raw_jpeg_data_left = []
raw_jpeg_data_right = []
cap = read_capture_file(sys.argv[1])

random.shuffle(cap)

FLAG_GOOD_DATA = 1 << 30
FLAG_GAZE_DATA = 1 << 0

def select_gaze(side):
    i = 0
    raw_jpeg_data = []
    all_labels = np.empty((1, len(cap), 17), dtype=np.float32)
    for frame in cap:
        _, _, routine_distance, routine_convergence, fov_adjust_distance, left_eye_pitch, left_eye_yaw, right_eye_pitch, right_eye_yaw, routine_left_lid, routine_right_lid, routine_brow_raise, routine_brow_angry, routine_widen, routine_squint, routine_dilate, routine_state = frame[0]
        if routine_state & FLAG_GOOD_DATA and routine_state & FLAG_GAZE_DATA:
            raw_jpeg_data.append([frame[1],frame[4][2][1],frame[4][1][1],frame[4][0][1]] if side == 'left' else [frame[2],frame[4][2][2],frame[4][1][2],frame[4][0][2]])
            all_labels[0][i] = frame[0]
            i = i + 1
    return babble_data.Loader(
        jpeg_datasets=[raw_jpeg_data,],
        dataset_probs=[1.0]
    ), all_labels, len(raw_jpeg_data)

def select_brow(side):
    i = 0
    e = 0
    raw_jpeg_data_squint = []
    raw_jpeg_data_other = []
    all_labels = np.empty((2, len(cap), 17), dtype=np.float32)
    for frame in cap:
        _, _, routine_distance, routine_convergence, fov_adjust_distance, left_eye_pitch, left_eye_yaw, right_eye_pitch, right_eye_yaw, routine_left_lid, routine_right_lid, routine_brow_raise, routine_brow_angry, routine_widen, routine_squint, routine_dilate, routine_state = frame[0]
        
        labels = list(frame[0])
        
        if routine_state & FLAG_GOOD_DATA and routine_brow_angry > 0.5:
            raw_jpeg_data_squint.append([frame[1],frame[4][2][1],frame[4][1][1],frame[4][0][1]] if side == 'left' else [frame[2],frame[4][2][2],frame[4][1][2],frame[4][0][2]])
            all_labels[0][i] = labels
            i = i + 1
        elif routine_state & FLAG_GOOD_DATA:
            raw_jpeg_data_other.append([frame[1],frame[4][2][1],frame[4][1][1],frame[4][0][1]] if side == 'left' else [frame[2],frame[4][2][2],frame[4][1][2],frame[4][0][2]])
            all_labels[1][e] = labels
            e = e + 1

    return babble_data.Loader(
        jpeg_datasets=[raw_jpeg_data_squint, raw_jpeg_data_other],
        dataset_probs=[0.5, 0.5]
    ), all_labels, len(raw_jpeg_data_squint)

def select_squint_wide_blink(side):
    i = 0
    e = 0
    raw_jpeg_data_squint = []
    raw_jpeg_data_other = []
    all_labels = np.empty((2, len(cap)*2, 17), dtype=np.float32)
    for frame in cap:
        _, _, routine_distance, routine_convergence, fov_adjust_distance, left_eye_pitch, left_eye_yaw, right_eye_pitch, right_eye_yaw, routine_left_lid, routine_right_lid, routine_brow_raise, routine_brow_angry, routine_widen, routine_squint, routine_dilate, routine_state = frame[0]

        labels = list(frame[0])

        o_lid = routine_left_lid

        if routine_squint > 0.5:
            labels[9] = 0.7
            labels[10] = 0.7
            routine_left_lid = routine_right_lid = 0.7
        elif routine_widen > 0.5:
            labels[9] = 0.0
            labels[10] = 0.0
            routine_left_lid = routine_right_lid = 0.0
        elif o_lid > 0.5:
            labels[9] = 0.3
            labels[10] = 0.3
            routine_left_lid = routine_right_lid = 0.3
        else:
            labels[9] = 1.0
            labels[10] = 1.0
            routine_left_lid = routine_right_lid = 1.0
        
        if routine_state & FLAG_GOOD_DATA and (routine_squint > 0.5 or routine_widen > 0.5 or o_lid < 0.5):
            raw_jpeg_data_squint.append([frame[1],frame[4][2][1],frame[4][1][1],frame[4][0][1]] if side == 'left' else [frame[2],frame[4][2][2],frame[4][1][2],frame[4][0][2]])

            labels[9] = 1.0
            labels[10] = 1.0
            if routine_squint > 0.5:
                labels[9] = 0.7
                labels[10] = 0.7
            elif routine_widen > 0.5:
                labels[9] = 0.0
                labels[10] = 0.0
            all_labels[0][i] = labels
            #all_labels[0][i+1] = labels
            i = i + 1
        elif routine_state & FLAG_GOOD_DATA:
            raw_jpeg_data_other.append([frame[1],frame[4][2][1],frame[4][1][1],frame[4][0][1]] if side == 'left' else [frame[2],frame[4][2][2],frame[4][1][2],frame[4][0][2]])#raw_jpeg_data_other.append([frame[1],frame[4][2][1],frame[4][1][1],frame[4][0][1]])

            labels[9] = 0
            labels[10] = 0
            all_labels[1][e] = labels
            #all_labels[1][e+1] = labels
            e = e + 1

    # invert gaze labels
    #all_labels[:, :, [9, 10]] = 1-all_labels[:, :, [9, 10]]

    return babble_data.Loader(
        jpeg_datasets=[raw_jpeg_data_squint, raw_jpeg_data_other],
        dataset_probs=[0.33333, 0.66666]
    ), all_labels, len(raw_jpeg_data_squint)


def select_squint_wide_brow(side):
    i = 0
    e = 0
    raw_jpeg_data_squint = []
    raw_jpeg_data_other = []
    all_labels = np.empty((2, len(cap), 17), dtype=np.float32)
    for frame in cap:
        _, _, routine_distance, routine_convergence, fov_adjust_distance, left_eye_pitch, left_eye_yaw, right_eye_pitch, right_eye_yaw, routine_left_lid, routine_right_lid, routine_brow_raise, routine_brow_angry, routine_widen, routine_squint, routine_dilate, routine_state = frame[0]
        
        labels = list(frame[0])
        
        if routine_state & FLAG_GOOD_DATA and (routine_squint > 0.5 or routine_widen > 0.5 or routine_brow_angry > 0.5 or (routine_left_lid < 0.5 and side == 'left') or (routine_right_lid < 0.5 and side == 'right')):
            raw_jpeg_data_squint.append([frame[1],frame[4][2][1],frame[4][1][1],frame[4][0][1]] if side == 'left' else [frame[2],frame[4][2][2],frame[4][1][2],frame[4][0][2]])
            all_labels[0][i] = labels
            i = i + 1
        elif routine_state & FLAG_GOOD_DATA:
            raw_jpeg_data_other.append([frame[1],frame[4][2][1],frame[4][1][1],frame[4][0][1]] if side == 'left' else [frame[2],frame[4][2][2],frame[4][1][2],frame[4][0][2]])
            all_labels[1][e] = labels
            e = e + 1

    # invert gaze labels
    #all_labels[:, :, [9, 10]] = 1-all_labels[:, :, [9, 10]]

    return babble_data.Loader(
        jpeg_datasets=[raw_jpeg_data_squint, raw_jpeg_data_other],
        dataset_probs=[0.5, 0.5]
    ), all_labels, len(raw_jpeg_data_squint)

print("Starting up babble data loader...")
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import random
import cv2
import onnx
import json

class AdapterWrapper(nn.Module):
    def __init__(self, mmchad):
        super(AdapterWrapper, self).__init__()
        self.mmchad = mmchad

    def forward(self, x):
        preds = self.mmchad(x)

        left_gaze_pitch = preds[0][0]
        left_gaze_yaw = preds[0][1]
        left_lid = preds[1][0]
        left_widen = preds[1][1]
        left_squeeze = 0
        left_brow = preds[2][0]

        right_gaze_pitch = preds[3][0]
        right_gaze_yaw = preds[3][1]
        right_lid = preds[4][0]
        right_widen = preds[4][1]
        right_squeeze = 0
        right_brow = preds[5][0]

        return [left_gaze_pitch, left_gaze_yaw, left_lid, left_widen, left_squeeze, left_brow,
                right_gaze_pitch, right_gaze_yaw, right_lid, right_widen, right_squeeze, right_brow]


def merge_models(names, sizes, output_names):
    modelsL = []
    modelsR = []

    all_names = []

    for side in ["left", "right"]:
        for name in output_names:
            for subname in name:
                all_names.append(side + subname)

    device = 'cpu'
    for i in range(len(names)):
        name = names[i]
        size = sizes[i]

        sdL = torch.load("./model_" + name + "_left.pth", weights_only=False, map_location=device)
        sdR = torch.load("./model_" + name + "_right.pth", weights_only=False, map_location=device)


        left = MicroChad(out_count=size).to(device)
        right = MicroChad(out_count=size).to(device)

        left.load_state_dict(sdL)
        right.load_state_dict(sdR)
        
        modelsL.append(left)
        modelsR.append(right)

    torch.onnx.export(
        MultiInputMergedMicroChad(modelsL, modelsR).cpu(),
        torch.randn(1, 8, 128, 128),
        sys.argv[2],
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    model = onnx.load(sys.argv[2])
    names_json = json.dumps(all_names)
    meta_prop = model.metadata_props.add()
    meta_prop.key = 'blendshape_names'
    meta_prop.value = names_json

    onnx.save(model, sys.argv[2])

TOTAL_STEPS_TRAINED = 0
TOTAL_STEPS_TRAINED_END = 0
LAST_SIM_EPOCH = 0

def train_model(kind, label_idx, class_count, steps, enable_gaze_correction=False, side='left', lr=0.001, batch_size=16, enable_noise_aug=False, do_warmup=True, do_cooldown=False):
    global TOTAL_STEPS_TRAINED
    global LAST_SIM_EPOCH

    if kind == 'brow':
        loader, all_labels, dataset_count = select_brow(side)
    if kind == 'sqwibl' or kind == 'blink':
        loader, all_labels, dataset_count = select_squint_wide_blink(side)
    if kind == 'sqwibrbl' or kind == "brbl":
        loader, all_labels, dataset_count = select_squint_wide_brow(side)
    elif kind == 'gaze':
        loader, all_labels, dataset_count = select_gaze(side)

    BATCH_SIZE = batch_size
    NUM_WORKERS = 16
    QUEUE_SIZE = 2048
    loader.start(NUM_WORKERS, QUEUE_SIZE)

    print("Setting up training...")

    batch_np = np.empty((BATCH_SIZE, 4, 128, 128), dtype=np.float32)

    # each value is (dataset, index)
    batch_idx = np.empty((BATCH_SIZE, 2), dtype=np.int64)

    print("Microchad init")

    model = MicroChad(out_count=class_count).to(device)

    print("optim init")
    o = torch.optim.AdamW(model.parameters(), lr=lr)

    def warmup_fn(step):
        x = step / (steps / 10)
        return min(1.0, (np.arctanh(1 - (x * 1.4 * np.pi) + 1) + 1) / 2)
        #return min(1.0, step / (steps / 10))  # Gradually increase LR for first 5 epochs

    def warmed_up_fn(step):
        return 1.0
    
    def cooldown_fn(step):
        x = step / steps
        return min(1.0, -np.arctanh(x / 1.4) + 1)

    warmup_scheduler = LambdaLR(o, lr_lambda=cooldown_fn if do_cooldown else warmup_fn if do_warmup else warmed_up_fn)

    print("\nTraining "+ side+" "+ kind+"...")
    #progress = tqdm(range(steps))
    all_L = 0
    total_images = 0
    tsteps = 0
    
    counts = np.zeros((2, ), dtype=np.int64)
    for i in range(steps):#for i in progress:

        # simulate old style printing
        #sim_epoch = int((TOTAL_STEPS_TRAINED / TOTAL_STEPS_TRAINED_END) * 100)
        #if sim_epoch != LAST_SIM_EPOCH:
        #    print("\n=== Epoch %d/%d ===\n" % (sim_epoch, 100), flush=True)
        #    LAST_SIM_EPOCH = sim_epoch

        #TOTAL_STEPS_TRAINED += i


        o.zero_grad()
        loader.fill_batch(batch_np, batch_idx)

        if True:
            set_mask = batch_idx[:, 0]
            label_mask = batch_idx[:, 1]

            labels = all_labels[set_mask, label_mask][:, label_idx]

            if enable_gaze_correction:
                labels =  (labels + 45) / 90

            #print(labels[0])

            labels = torch.tensor(labels, device=device)
            inputs = torch.tensor(batch_np, device=device)

            if enable_noise_aug:
                if random.random() > 0.8:
                    labels = torch.rand_like(labels)
                    inputs = torch.rand_like(inputs)
                    is_rand_batch = True
                else:
                    labels += torch.randn_like(labels) * 0.1
                    inputs += torch.randn_like(inputs) * 0.1
                    

                    is_rand_batch = False
                    
            else:
                is_rand_batch = False
            tsteps+=1
            labels = torch.clip(labels, 0, 1)
            inputs = torch.clip(inputs, 0, 1)
            loss = F.mse_loss(model(inputs), labels)

            if not is_rand_batch:
                all_L += loss.detach()
            total_images += BATCH_SIZE

            loss.backward()
            o.step()
            warmup_scheduler.step()

            print("\rBatch %u/%u, Loss: %.6f" % (i, steps, float(loss)), flush=True)

        if i % 100 == 0 and False:  # Update the preview window every 100 steps
            GRID_SIZE = 4
            PREVIEW_COUNT = GRID_SIZE * GRID_SIZE
            
            # Select the first PREVIEW_COUNT images and labels
            # Use inputs.cpu().numpy() to get the images, as batch_np might not be the final input if is_rand_batch was True
            preview_images_np = inputs[:PREVIEW_COUNT].cpu().numpy() # Shape (16, 4, 128, 128)
            preview_labels = labels[:PREVIEW_COUNT].cpu().numpy()
            
            rows = []
            
            for r in range(GRID_SIZE):
                row_images = []
                for c in range(GRID_SIZE):
                    idx = r * GRID_SIZE + c
                    
                    # Get image and label
                    img = preview_images_np[idx] # (4, 128, 128)
                    label_val = preview_labels[idx]
                    
                    # Convert from C, H, W to H, W, C, select first 3 channels, scale to 0-255, and convert to uint8
                    # Assumes input images in inputs are in the 0.0 to 1.0 range (like the labels clip)
                    display_img = img[:3] # (3, 128, 128)
                    display_img = np.transpose(display_img, (1, 2, 0)) # (128, 128, 3)
                    
                    # Scale float (0.0 to 1.0) to uint8 (0 to 255)
                    display_img = (np.clip(display_img * 255, 0, 255)).astype(np.uint8) 
                    
                    # Convert RGB (assumed) to BGR for cv2 display
                    display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)

                    # Draw label text. Format label as a string with 4 decimal places.
                    label_text = f"{label_val.item():.4f}" # .item() for scalar numpy/torch tensor
                    
                    # Draw text on the image
                    # Position (5, 20) is near top-left. Color is bright green (0, 255, 0) BGR.
                    cv2.putText(
                        display_img, 
                        label_text, 
                        (5, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (0, 255, 0), 
                        2, 
                        cv2.LINE_AA
                    )
                    
                    row_images.append(display_img)
                    
                # Concatenate images in the row horizontally
                rows.append(cv2.hconcat(row_images))
                
            # Concatenate rows vertically to form the final grid
            grid_image = cv2.vconcat(rows)
            
            # Display the grid
            cv2.imshow("Training Input & Target Label Preview (4x4 Grid)", grid_image)
            # cv2.waitKey(1) allows the window to update and keeps the program responsive
            cv2.waitKey(1) 
        # ----------------------------------------------------------------------
        
        if tsteps % 50 == 49:
            #progress.set_description("loss:%.4f image:%d epoch:%d lr:%.5f" % (all_L / 50, total_images, total_images / dataset_count, warmup_scheduler.get_last_lr()[0]))
            all_L = 0

    loader.stop()
    torch.save(model.state_dict(), "./model_" + kind + "_" + side+".pth")
    
    del loader
    
    # NEW: Close the OpenCV window when training is finished
    cv2.destroyAllWindows()

print("\n=== Epoch %d/%d ===\n" % (1, 6), flush=True)
start = time.time()
train_model("gaze", [7, 8], 2, 1000, enable_gaze_correction=True, enable_noise_aug=False, batch_size=128, do_warmup=False, side='right', do_cooldown=True)
print("\nEpoch %d/%d completed in %.2fs. Average loss: %.6f\n" % (1,6, time.time() - start, 0), flush=True)
start = time.time()
print("\n=== Epoch %d/%d ===\n" % (2, 6), flush=True)
train_model("gaze", [5, 6], 2, 1000, enable_gaze_correction=True, enable_noise_aug=False, batch_size=128, do_warmup=False, side='left', do_cooldown=True)
print("\nEpoch %d/%d completed in %.2fs. Average loss: %.6f\n" % (2,6, time.time() - start, 0), flush=True)
start = time.time()
print("\n=== Epoch %d/%d ===\n" % (3, 6), flush=True)
train_model("blink", [10,13], 2, 1600, enable_gaze_correction=False, enable_noise_aug=True, batch_size=16, side='right', lr=5e-5)
print("\nEpoch %d/%d completed in %.2fs. Average loss: %.6f\n" % (3,6, time.time() - start, 0), flush=True)
start = time.time()
print("\n=== Epoch %d/%d ===\n" % (4, 6), flush=True)
train_model("blink", [9,13], 2, 1600, enable_gaze_correction=False, enable_noise_aug=True, batch_size=16, side='left', lr=5e-5)
print("\nEpoch %d/%d completed in %.2fs. Average loss: %.6f\n" % (4,6, time.time() - start, 0), flush=True)
start = time.time()
print("\n=== Epoch %d/%d ===\n" % (5, 6), flush=True)
train_model("brow", [12,], 1, 1600, enable_gaze_correction=False, side='right')e
print("\nEpoch %d/%d completed in %.2fs. Average loss: %.6f\n" % (5,6, time.time() - start, 0), flush=True)
start = time.time()
print("\n=== Epoch %d/%d ===\n" % (6, 6), flush=True)
train_model("brow", [12,], 1, 1600, enable_gaze_correction=False, side='left')
print("\nEpoch %d/%d completed in %.2fs. Average loss: %.6f\n" % (6,6, time.time() - start, 0), flush=True)


TOTAL_STEPS_TRAINED_END = 1000 + 1000 + 1600 + 1600 + 1600 + 1600

merge_models(["gaze", "blink", "brow"], [2, 2, 1], [["EyePitch", "EyeYaw"], ["EyeLid", "EyeWiden"], ["Brow"]])

print("\nTraining completed successfully!\n", flush=True)









