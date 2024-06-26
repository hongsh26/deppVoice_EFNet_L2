import matplotlib.pyplot as plt
import os
import shutil
import librosa
import librosa.display
from dotenv import load_dotenv
from sklearn.utils import shuffle
import torchvision
import torch
from PIL import Image
import numpy as np

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)
#train / test classification
files = os.listdir(os.getenv('dir_fake'))
count = 0
for file in files:
    # if(count%2 == 0):
    src = os.path.join(os.getenv('dir_fake'), file)
    dst = os.path.join(os.getenv('test_fake_dir'), file)
    shutil.move(src, dst)
    count += 1

#making_spectrogram function
def save_spectrogram(path, save):
    audio, sample_rate = librosa.load(path)
    n_fft = 2048  # 창 크기
    hop_length = n_fft // 4  # 홉 크기
    S = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=128)
    mel_spect = librosa.power_to_db(S, ref=np.max)

    dir_path = save
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_name = os.path.basename(path).replace('.wav', '.png')
    save_path = os.path.join(dir_path, file_name)

    plt.figure(figsize=(10, 4))
    img = librosa.display.specshow(mel_spect, sr=sample_rate, x_axis='time', y_axis='log')
    plt.colorbar(img, format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.tight_layout()
    print(save_path)
    plt.savefig(save_path)
    plt.close()

#processing each file
def preprocessing(set, save):
    files = os.listdir(set)
    for file in files:
        print(os.path.join(set, file))
        save_spectrogram(os.path.join(set, file), save)

def shuffle_dataset(fake_dir, real_dir,train_test):

    # Get the list of files in each directory
    fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if os.path.isfile(os.path.join(fake_dir, f))]
    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if os.path.isfile(os.path.join(real_dir, f))]

    # Label the files and combine them into a single list
    labeled_data = [(f, 'fake') for f in fake_files] + [(f, 'real') for f in real_files]

    # Separate paths and labels
    paths, labels = zip(*labeled_data)

    # Shuffle the paths and labels
    shuffled_paths, shuffled_labels = shuffle(paths, labels)

    # Check the result
    for path, label in zip(shuffled_paths[:10], shuffled_labels[:10]):  # Print the first 10 elements as a sample
        print(f"Path: {path}, Label: {label}")

    # Optionally, move the files to a new directory structure if needed
    if(train_test == 0):
        output_dir = '/Users/hongseunghyuk/PycharmProjects/practice 1/data/train_set/train_shuffle'
    else:
        output_dir = '/Users/hongseunghyuk/PycharmProjects/practice 1/data/test_set/test_shuffle'
    os.makedirs(output_dir, exist_ok=True)
    for i, (path, label) in enumerate(zip(shuffled_paths, shuffled_labels)):
        # Define new path for each file in the output directory
        new_path = os.path.join(output_dir, f"{i}_{label}_{os.path.basename(path)}")
        shutil.copy2(path, new_path)

    print("Files have been copied and labeled successfully.")


def load_data(data_dir):
    data = []
    labels = []
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.Resize((28, 28)),
        torchvision.transforms.ToTensor()
    ])
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        image = Image.open(filepath)
        image = transform(image)
        data.append(image)
        label = 1 if 'fake' in filename else 0
        labels.append(label)
    return torch.stack(data), torch.tensor(labels, dtype=torch.long)


print("Train start -------------------------")
#Preprocessing -> Train Fake
preprocessing(os.getenv('train_fake_dir'), os.getenv('train_fake_spectrogram'))
#Preprocessing -> Train Real
preprocessing(os.getenv('train_real_dir'), os.getenv('train_real_spectrogram'))

print("Test start ---------------------------")
# Preprocessing -> Test Fake
preprocessing(os.getenv('test_fake_dir'), os.getenv('test_fake_spectrogram'))
# Preprocessing -> Test Real
preprocessing(os.getenv('test_real_dir'), os.getenv('test_real_spectrogram'))

#shuffle Train set
print("shuffle ---------------------")
shuffle_dataset(os.getenv('train_fake_spectrogram'),os.getenv('train_real_spectrogram'),0)
shuffle_dataset(os.getenv('test_fake_spectrogram'),os.getenv('test_real_spectrogram'),1)
train_data, train_labels = load_data(os.getenv('train_shuffle'))
test_data, test_labels = load_data(os.getenv('test_shuffle'))
torch.save((train_data, train_labels),'train_data.pt')
torch.save((train_data, train_labels),'test_data.pt')