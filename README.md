# Final-year-project---1
Implemented deep learning and preprocessing  techniques to identify synthetic streaming-based media,   including video, images. Successfully distinguished   between fake and real content, providing explanations for  the identification process. 



Image:
#random selecting
import os
import random
import shutil
def select_random_images_from_folders(folder_paths, num_images):
 selected_images = []
 for folder_path in folder_paths:
 image_files = [file for file in os.listdir(folder_path) if 
file.lower().endswith(('.png', '.jpg', '.jpeg'))]
 selected_images.extend(random.sample(image_files, min(num_images, 
len(image_files))))
 return selected_images
 
# Path to the original dataset
dataset_path = r"E:\video dataset"

# Path to the new folder to save the selected images
selected_images_folder = r"E:\selected_images"

# Create the new folders for "fake" and "real" images
os.makedirs(os.path.join(selected_images_folder, "fake"), exist_ok=True)
os.makedirs(os.path.join(selected_images_folder, "real"), exist_ok=True)

# Number of images to select from each subfolder
num_images_per_folder = 5000

# Path to the folders containing images in the "fake" directory36
fake_folders = [os.path.join(dataset_path, "fake")]

# Path to the folders containing images in the "real" directory
real_folders = [os.path.join(dataset_path, "real")]

# Select random images from folders in the "fake" directory
selected_fake_images = select_random_images_from_folders(fake_folders, 
num_images_per_folder)

# Select random images from folders in the "real" directory
selected_real_images = select_random_images_from_folders(real_folders, 
num_images_per_folder)

# Copy the selected images to the new "fake" folder
for image in selected_fake_images:
 image_path = os.path.join(dataset_path, "fake", image)
 shutil.copy(image_path, os.path.join(selected_images_folder, "fake"))
 
# Copy the selected images to the new "real" folder
for image in selected_real_images:
 image_path = os.path.join(dataset_path, "real", image)
 shutil.copy(image_path, os.path.join(selected_images_folder, "real"))
 
#preprocessing
import cv2
import os
def preprocess_image(image_path, target_size=(299, 299)):

 # Load image using OpenCV
 image = cv2.imread(image_path)
 if image is None:
  print(f"Error loading image: {image_path}")
return None

 # Resize image37
 image = cv2.resize(image, target_size)
 
 # Convert image to RGB (OpenCV uses BGR by default)
 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
 # Normalize pixel values to [0, 1]
 image = image.astype('float32') / 255.0
 return image
def preprocess_images_in_directory(input_dir, output_dir):

 # Iterate through all directories and subdirectories
 for root, dirs, files in os.walk(input_dir):
 for file in files:
           if file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
           
# Construct full path to the image file
image_path = os.path.join(root, file)

# Preprocess the image
preprocessed_image = preprocess_image(image_path)
if preprocessed_image is not None:

# Create the corresponding directory structure in the output directory
output_subdir = os.path.join(output_dir, os.path.relpath(root, input_dir))
os.makedirs(output_subdir, exist_ok=True)

# Save the preprocessed image to the output directory
output_path = os.path.join(output_subdir, file)
 cv2.imwrite(output_path, cv2.cvtColor((preprocessed_image * 
255).astype('uint8'), cv2.COLOR_RGB2BGR))
 else:
 print(f"Error preprocessing image: {image_path}")
 
# Input directory containing raw images
raw_images_dir = r"E:\video dataset\fake"

# Output directory to save preprocessed images38
output_dir = r"E:\preprocessed fake images"
# Preprocess images in the input directory and save them to the output directory
preprocess_images_in_directory(raw_images_dir, output_dir)

#splitting the dataset
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
def load_and_split_dataset(input_dir, output_dir, train_size=0.7, val_size=0.1, 
test_size=0.2):

 # Function to load the dataset and split it into train, validation, and test sets
 # Create the output directory if it doesn't exist
 os.makedirs(output_dir, exist_ok=True)
 images = []
 labels = []
 
 # Iterate through the 'real' and 'fake' folders separately
 for label in ['real', 'fake']:
    label_dir = os.path.join(input_dir, label)
if not os.path.isdir(label_dir):
print(f"Directory '{label}' not found.")
continue
       for file in os.listdir(label_dir):
if file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
image_path = os.path.join(label_dir, file)

# Load the image
image = cv2.imread(image_path)
images.append(image)

# Determine the label for the image39
 labels.append(label)
 
 # Convert lists to NumPy arrays
 images = np.array(images)
 labels = np.array(labels)
 
 # Split the dataset into train, validation, and test sets
 train_images, test_images, train_labels, test_labels = train_test_split(images, labels, 
test_size=test_size, stratify=labels, random_state=42)
 train_images, val_images, train_labels, val_labels = train_test_split(train_images, 
train_labels, test_size=val_size / (1 - test_size), stratify=train_labels, 
random_state=42)

 # Save train images along with their labels
 save_images(os.path.join(output_dir, 'train'), train_images, train_labels)
 
 # Save validation images along with their labels
 save_images(os.path.join(output_dir, 'val'), val_images, val_labels)
 
 # Save test images along with their labels
 save_images(os.path.join(output_dir, 'test'), test_images, test_labels)
def save_images(directory, images, labels):

 # Function to save images along with their labels
 for image, label in zip(images, labels):
    class_dir = os.path.join(directory, label)
os.makedirs(class_dir, exist_ok=True)
image_filename = os.path.join(class_dir, f'{len(os.listdir(class_dir)) + 1}.jpg')
cv2.imwrite(image_filename, image)

# Define the input and output directories
input_dir = r"E:\dfdc_subfolder" # Modify this to your actual input directory
output_dir = r"E:\splitted image dataset" # Modify this to your desired output 
directory

# Load and split the dataset with the specified ratios
load_and_split_dataset(input_dir, output_dir, train_size=0.7, val_size=0.1, 
test_size=0.2)

#model
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
def analyze_xception_architecture():

 # Load XceptionNet without the top (classification) layer
 base_model = tf.keras.applications.Xception(weights='imagenet', 
include_top=False)

 # Print a summary of the model architecture
 base_model.summary()
def modified_xception(input_shape=(299, 299, 3), num_classes=1, start_index=80):
 """
 Create a modified XceptionNet model with strategically replaced layers.
 """
 
 # Load XceptionNet without the top (classification) layer
 base_model = tf.keras.applications.Xception(weights='imagenet', 
include_top=False, input_shape=input_shape)

 # Freeze the base model layers
 base_model.trainable = False
 
 # Get the layers of the base model
 layers = [layer for layer in base_model.layers]41
 
 # Replace specific layers with depthwise separable convolutions
 for i in range(start_index, len(layers)):
   layer = layers[i]
if isinstance(layer, tf.keras.layers.Conv2D):

# Replace standard Conv2D with DepthwiseConv2D followed by 1x1 
Conv2D
 x = tf.keras.layers.DepthwiseConv2D((3, 3), activation='relu', 
padding='same')(layer.output)
 x = tf.keras.layers.Conv2D(layer.filters, (1, 1), activation='relu', 
padding='same')(x)
 layers[i] = tf.keras.models.Model(inputs=base_model.input, 
outputs=x).layers[-1] # Update the modified layer

 # Create the modified model
 modified_model = tf.keras.models.Model(inputs=base_model.input, 
outputs=layers[-1].output)

 # Add classification layers
 x = tf.keras.layers.GlobalAveragePooling2D()(modified_model.output)
 predictions = tf.keras.layers.Dense(num_classes, activation='sigmoid')(x)
 model = tf.keras.models.Model(inputs=modified_model.input, outputs=predictions)
 return model
 
# Create the modified XceptionNet model with custom starting index
model = modified_xception(start_index=100)

# Unfreeze some layers for fine-tuning
for layer in model.layers[:100]:
 layer.trainable = False
for layer in model.layers[100:]:
 layer.trainable = True
 
# Compile the model with gradient clipping42
opt = tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=0.5) # You can adjust 
the clipvalue as needed
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# Define directories
base_dir = r"E:\splitted image dataset" # Change this to your dataset directory
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

# Define image dimensions and batch size
image_size = (299, 299)
batch_size = 64

# Create data generators for train and validation sets
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
 train_dir,
 target_size=image_size,
 batch_size=batch_size,
 class_mode='binary', # Adjust class mode based on your dataset
 shuffle=True
)
val_generator = val_datagen.flow_from_directory(
 val_dir,
 target_size=image_size,
 batch_size=batch_size,
 class_mode='binary', # Adjust class mode based on your dataset
 shuffle=False # No need to shuffle validation data
)

# Define callbacks43
early_stopping = EarlyStopping(monitor='val_loss', patience=5, 
restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, 
min_lr=1e-6)

# Train the model
history = model.fit(
 train_generator,
 steps_per_epoch=len(train_generator),
 epochs=1, # Adjust number of epochs as needed
 validation_data=val_generator,
 validation_steps=len(val_generator),
 callbacks=[early_stopping, reduce_lr]
)

# Save the trained model
model.save(r"E:\modified_xception_model.h5")

# Print training history
print("Training History:")
print(history.history)

# Print final training and validation accuracy along with loss
print("Final Training Accuracy:", history.history['accuracy'][-1])
print("Final Training Loss:", history.history['loss'][-1])
print("Final Validation Accuracy:", history.history['val_accuracy'][-1])
print("Final Validation Loss:", history.history['val_loss'][-1])

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(val_generator)

# Predict the validation set labels
y_val_pred = model.predict(val_generator)
y_val_true = val_generator.classes44

# Calculate precision, recall, and F1-score for the validation set
val_precision = precision_score(y_val_true, y_val_pred > 0.5)
val_recall = recall_score(y_val_true, y_val_pred > 0.5)
val_f1 = f1_score(y_val_true, y_val_pred > 0.5)

# Print the evaluation metrics for the validation set
print("Validation Precision:", val_precision)
print("Validation Recall:", val_recall)
print("Validation F1-score:", val_f1)

# Plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(['Training', 'Validation'], [history.history['accuracy'][-1], 
history.history['val_accuracy'][-1]], label='Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.ylim(0, 1) # Limit y-axis to range [0, 1] for accuracy
plt.legend()

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.bar(['Training', 'Validation'], [history.history['loss'][-1], history.history['val_loss'][-
1]], label='Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()
#test code
image_size = (299, 299)
batch_size = 32
test_dir=r"E:\splitted image dataset1\test"45
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
 test_dir,
 target_size=image_size,
 batch_size=batch_size,
 class_mode='binary', 
 shuffle=False 
)
test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator))
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
Video:
#To select random videos
import os
import random
import shutil

# Path to the original DFDC dataset
dfdc_path = r"D:\dfdc_train_part_04"

# Path to the new folder to save the selected videos
dfdc_subfolder = r"E:\dfdc_subfolder1"

# Create the new folder if it doesn't exist
os.makedirs(dfdc_subfolder, exist_ok=True)

# Create "fake" folder within selected_videos_path
fake_folder_path = os.path.join(dfdc_subfolder, "fake")
os.makedirs(fake_folder_path, exist_ok=True)46

# Number of videos to select from the "fake" folder
num_videos_per_folder = 1180

# Function to select random videos from a folder
def select_random_videos(folder_path, num_videos):

 # List all video files in the folder
 video_files = os.listdir(folder_path)
 
 # Randomly select num_videos videos
 selected_videos = random.sample(video_files, num_videos)
 return selected_videos
 
# Select random videos from the "fake" folder
fake_folder_path_original = os.path.join(dfdc_path, "fake")
selected_fake_videos = select_random_videos(fake_folder_path_original, 
num_videos_per_folder)

# Copy the selected videos to the new "fake" folder
for video in selected_fake_videos:
 video_path = os.path.join(fake_folder_path_original, video)
 shutil.copy(video_path, fake_folder_path)
 
# Print the number of selected videos
print("Number of videos sampled from fake folder:", len(selected_fake_videos))
#video to framesimport cv2
import os
import numpy as np
def calculate_brightness(image):
 gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 brightness = np.mean(gray)
 return brightness47
def extract_frames(video_path, output_dir, brightness_threshold=50):
 cap = cv2.VideoCapture(video_path)
 frame_count = 0
 while cap.isOpened():
       ret, frame = cap.read()
if not ret:
break
frame_count += 1
brightness = calculate_brightness(frame)
if brightness > brightness_threshold:
frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg") # 
Add leading zeros for frame number
     try:
cv2.imwrite(frame_path, frame)
print(f"Frame extracted and saved: {frame_path}")
except Exception as e:
print(f"Error saving frame: {e}")
 cap.release()
input_dir = r"E:\dfdc_subfolder1"
output_dir = r"E:\dfdc_frames1" 
brightness_threshold = 50
for folder_name in ["real", "fake"]:
 folder_path = os.path.join(input_dir, folder_name)
 if os.path.isdir(folder_path):
        output_folder = os.path.join(output_dir, folder_name)
os.makedirs(output_folder, exist_ok=True)
for video_file in os.listdir(folder_path):
video_path = os.path.join(folder_path, video_file)
output_video_dir = os.path.join(output_folder, os.path.splitext(video_file)[0])
os.makedirs(output_video_dir, exist_ok=True)
extract_frames(video_path, output_video_dir, brightness_threshold)
print(f"Processed video: {video_path}")

#Face extrcation
import os
import cv2
import numpy as np
import face_recognition
def preprocess_frame(frame, target_size=(224, 224), normalize=True):

 # Convert frame to RGB (face_recognition uses RGB)
 rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
 # Locate faces in the frame
 face_locations = face_recognition.face_locations(rgb_frame)
 if len(face_locations) > 0:
 
 # Assuming only one face is detected, you can modify this according to your 
needs
 top, right, bottom, left = face_locations[0]
 
  # Crop the frame to include only the detected face region
cropped_frame = frame[top:bottom, left:right]

  # Resize the cropped frame to the target size
resized_frame = cv2.resize(cropped_frame, target_size)
       # Normalize pixel values if required
if normalize:

# Convert pixel values to [0, 1] range
normalized_frame = resized_frame.astype(np.float32) / 255.0
return normalized_frame
else:
return resized_frame
 else:
 
 # If no faces are detected, return None
 return None49
def preprocess_dataset(input_dir, output_dir, target_size=(224, 224), 
normalize=True):

 # Iterate over all directories and subdirectories
 for root, dirs, files in os.walk(input_dir):
    for filename in files:
if filename.endswith('.jpg') or filename.endswith('.png'):

# Read the frame
frame = cv2.imread(os.path.join(root, filename))

 # Preprocess the frame
 preprocessed_frame = preprocess_frame(frame, target_size=target_size, 
normalize=normalize)
  structure
      if preprocessed_frame is not None:
      
# Construct the output directory path mirroring the input directory 
relative_path = os.path.relpath(root, input_dir)
output_subdir = os.path.join(output_dir, relative_path)
os.makedirs(output_subdir, exist_ok=True)

# Save the preprocessed frame to the output directory
output_filename = os.path.join(output_subdir, filename)
cv2.imwrite(output_filename, preprocessed_frame * 255.0 if normalize 
else preprocessed_frame)

# Example usage:
input_directory = r"D:\dfdc_frames1" # Replace with the path to your input dataset 
directory
output_directory = r"E:\video dataset" # Replace with the path to your output 
directory
preprocess_dataset(input_directory, output_directory)
#splitting the dataset
import os50
import shutil
import random
def split_folders(input_dir, output_dir, train_ratio=0.7, test_ratio=0.2, val_ratio=0.1):

 # Create output directories if they don't exist
 for split in ['train', 'test', 'val']:
  os.makedirs(os.path.join(output_dir, split, 'real'), exist_ok=True)
os.makedirs(os.path.join(output_dir, split, 'fake'), exist_ok=True)
 for label in ['real', 'fake']:
   label_dir = os.path.join(input_dir, label)
folders = os.listdir(label_dir)
random.shuffle(folders)
   total_folders = len(folders)
train_end = int(total_folders * train_ratio)
test_end = int(total_folders * (train_ratio + test_ratio))
   train_folders = folders[:train_end]
test_folders = folders[train_end:test_end]
val_folders = folders[test_end:]
    for folder in train_folders:
source_dir = os.path.join(label_dir, folder)
dest_dir = os.path.join(output_dir, 'train', label, folder)
shutil.copytree(source_dir, dest_dir)
    for folder in test_folders:
source_dir = os.path.join(label_dir, folder)
dest_dir = os.path.join(output_dir, 'test', label, folder)
shutil.copytree(source_dir, dest_dir)
  for folder in val_folders:
source_dir = os.path.join(label_dir, folder)51
  dest_dir = os.path.join(output_dir, 'val', label, folder)
shutil.copytree(source_dir, dest_dir)

# Define input and output directories
input_directory = r"E:\video dataset"
output_directory = r"E:\splitted video dataset"

# Split folders into train, test, and val sets
split_folders(input_directory, output_directory)
#feature extraction import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

# Function to load and preprocess images
def load_and_preprocess_image(image_path):
 image = cv2.imread(image_path)
 if image is None:
  print(f"Error: Unable to load image from {image_path}. Skipping...")
return None
 preprocessed_image = preprocess_input(image)
 return preprocessed_image
 
# Load pre-trained EfficientNetB0 model with weights from 'imagenet'
base_model = EfficientNetB0(weights='imagenet', include_top=False)

# Freeze base layers (freeze the first 70% of layers)
freeze_index = int(len(base_model.layers) * 0.7)
for layer in base_model.layers[:freeze_index]:
 layer.trainable = False
 
# Function to extract features from images using the pre-trained model52
def extract_features_from_images(input_dir, output_dir, model):
 for dataset_folder in ["train"]: 
 print(f"Processing dataset folder: {dataset_folder}")
  dataset_input_dir = os.path.join(input_dir, dataset_folder)
dataset_output_dir = os.path.join(output_dir, dataset_folder)
   if not os.path.exists(dataset_input_dir):
print(f"Dataset input directory not found: {dataset_input_dir}")
continue
  if not os.path.exists(dataset_output_dir):
os.makedirs(dataset_output_dir)
  for class_folder in ["real", "fake"]:
print(f"Processing class folder: {class_folder}")
  class_input_dir = os.path.join(dataset_input_dir, class_folder)
class_output_dir = os.path.join(dataset_output_dir, class_folder)
   if not os.path.exists(class_input_dir):
print(f"Class input directory not found: {class_input_dir}")
continue
  if not os.path.exists(class_output_dir):
os.makedirs(class_output_dir)
    # Recursively iterate through subfolders
for root, dirs, files in os.walk(class_input_dir):

# Create corresponding output subdirectories
output_subdir = os.path.join(class_output_dir, os.path.relpath(root, 
class_input_dir))
 os.makedirs(output_subdir, exist_ok=True)53
    for image_file in files:
if image_file.endswith('.jpg') or image_file.endswith('.png'):
image_path = os.path.join(root, image_file)
print(f"Processing image: {image_path}")

    # Load and preprocess the image
preprocessed_image = load_and_preprocess_image(image_path)
if preprocessed_image is None:
continue
  axis=0))
  
# Extract features using EfficientNet
features = model.predict(np.expand_dims(preprocessed_image, 

 # Construct output file path
 output_file = os.path.join(output_subdir, 
f"{image_file.split('.')[0]}_features.npy")

   # Save features to file
np.save(output_file, features)
print(f"Features extracted and saved for image: {image_file}")

# Example usage:
input_directory = r"E:\splitted video dataset" # Replace with the path to your dataset 
directory
output_directory = r"E:\extractedFeatures" # Replace with the desired output 
directory
extract_features_from_images(input_directory, output_directory, base_model)

#custom LSTM model training the model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import Callback, EarlyStopping54
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
class CustomGradientClipping(Callback):
 def __init__(self, clip_value, lstm_units):
   super(CustomGradientClipping, self).__init__()
self.clip_value = clip_value
self.lstm_units = lstm_units
 def on_batch_end(self, batch, logs=None):
  if logs is None or 'inputs' not in logs or 'targets' not in logs:
return
 inputs, targets = logs['inputs'], logs['targets']
    with tf.GradientTape() as tape:
predictions = self.model(inputs, training=True)
loss = self.model.compiled_loss(targets, predictions,
regularization_losses=self.model.losses)
 gradients = tape.gradient(loss, self.model.trainable_variables)
     lstm_layer = self.model.get_layer('lstm')
lstm_weights = lstm_layer.get_weights()
recurrent_kernel = lstm_weights[1]
_, _, c_t, _ = tf.split(recurrent_kernel, 4, axis=0)
c_t = tf.transpose(c_t)
  forget_gate_gradients = tf.matmul(gradients, c_t)
input_gate_gradients = gradients
 forget_gate_threshold = 0.555
 input_gate_threshold = 0.5
 clipped_forget_gate_gradients = tf.clip_by_value(forget_gate_gradients, -
forget_gate_threshold,
 forget_gate_threshold)
 clipped_input_gate_gradients = tf.clip_by_value(input_gate_gradients, -
input_gate_threshold,
 input_gate_threshold)
 adaptive_clip_factor = 1.5
 adaptive_clip = self.clip_value * tf.reduce_mean(tf.abs(c_t)) / (self.lstm_units * 
adaptive_clip_factor)
 clipped_forget_gate_gradients = tf.clip_by_value(clipped_forget_gate_gradients, 
-adaptive_clip, adaptive_clip)
 clipped_input_gate_gradients = tf.clip_by_value(clipped_input_gate_gradients, -
adaptive_clip, adaptive_clip)
 gradients = gradients * 0.25
 gradients += tf.concat([clipped_input_gate_gradients, 
clipped_forget_gate_gradients], axis=0)
 self.model.optimizer.apply_gradients(zip(gradients, 
self.model.trainable_variables))

# Define dimensions of input data
num_timesteps = 7
num_features = 1280
batch_size = 32

# Define hyperparameters
num_units = 1280
num_classes = 2
epochs = 1
clip_value = 0.556

# Define total_samples
total_samples = 0

# Create LSTM model
model = Sequential([
 LSTM(units=num_units, input_shape=(num_timesteps, num_features), 
name='lstm', return_sequences=False),
 Dense(units=1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
def count_files(directory):
 count = 0
 for root, dirs, files in os.walk(directory):
 count += len(files)
 return count
def data_generator(real_directory, fake_directory, batch_size):
 all_real_files = []
 for root, dirs, files in os.walk(real_directory):
   for file in files:
if file.endswith('.npy'):
all_real_files.append(os.path.join(root, file))
 all_fake_files = []
 for root, dirs, files in os.walk(fake_directory):
   for file in files:
if file.endswith('.npy'):
all_fake_files.append(os.path.join(root, file))
 all_files = all_real_files + all_fake_files
 np.random.shuffle(all_files)57
 data = np.zeros((batch_size, num_timesteps, num_features))
 labels = np.zeros((batch_size,))
 index = 0
 for file in all_files:
     features = np.load(file)
if features is not None:

# Assuming channel dimension is at index 2
features = features[:, :, 0, :]
features = features[0, :num_timesteps, :] # Truncate or pad the sequence

    # Pad features if necessary
if features.shape[0] < num_timesteps:
pad_width = ((0, num_timesteps - features.shape[0]), (0, 0))
features = np.pad(features, pad_width, mode='constant', constant_values=0)
 data[index] = features
 
     # Extract label from directory structure
if "real" in file:
labels[index] = 0 # Real label
elif "fake" in file:
labels[index] = 1 # Fake label
 index += 1
     if index == batch_size:
yield data, labels
data = np.zeros((batch_size, num_timesteps, num_features))
labels = np.zeros((batch_size,))
index = 0

 # Yield remaining data if batch size doesn't evenly divide total samples58
 if index > 0:
 yield data[:index], labels[:index]
 
# Directory paths for training and validation data
train_real_dir = r"E:\extractedFeatures\train\real"
train_fake_dir = r"E:\extractedFeatures\train\fake"
val_real_dir = r"E:\extractedFeatures\val\real"
val_fake_dir = r"E:\extractedFeatures\val\fake"

# Count number of samples
train_real_samples = count_files(train_real_dir)
train_fake_samples = count_files(train_fake_dir)
val_real_samples = count_files(val_real_dir)
val_fake_samples = count_files(val_fake_dir)

# Load train and validation data generators
train_data_generator = data_generator(train_real_dir, train_fake_dir, batch_size)
val_data_generator = data_generator(val_real_dir, val_fake_dir, batch_size)

# Calculate total real and fake samples
total_real_samples = train_real_samples + val_real_samples
total_fake_samples = train_fake_samples + val_fake_samples

# Instantiate custom gradient clipping callback
custom_gradient_clipper = CustomGradientClipping(clip_value=clip_value, 
lstm_units=num_units)

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, 
restore_best_weights=True)

# Count number of samples
train_samples = train_real_samples + train_fake_samples
val_samples = val_real_samples + val_fake_samples59
steps_per_epoch = train_samples // batch_size
validation_steps = val_samples // batch_size

# Calculate class weights
class_weights = {
 0: total_fake_samples / (total_real_samples + total_fake_samples), # Weight for 
class 0 (real samples)
 1: total_real_samples / (total_real_samples + total_fake_samples) # Weight for 
class 1 (fake samples)
}

# Train the model with entire dataset and validate using validation data
history = model.fit(
 train_data_generator,
 steps_per_epoch=train_samples // batch_size,
 epochs=epochs,
 validation_data=val_data_generator,
 validation_steps=val_samples // batch_size,
 callbacks=[early_stopping]
)

# Print training history
print("Training History:")
print(history.history)

# Print final training and validation accuracy along with loss
print("Final Training Accuracy:", history.history['accuracy'][-1])
print("Final Training Loss:", history.history['loss'][-1])
print("Final Validation Accuracy:", history.history['val_accuracy'][-1])
print("Final Validation Loss:", history.history['val_loss'][-1])

# Evaluate on validation data60
val_data_generator = data_generator(val_real_dir, val_fake_dir, 
batch_size=val_samples)
val_data, val_labels = next(val_data_generator)
val_predictions = model.predict(val_data)
val_predictions_binary = (val_predictions > 0.5).astype(int)

# Calculate evaluation metrics
precision = precision_score(val_labels, val_predictions_binary)
recall = recall_score(val_labels, val_predictions_binary)
f1 = f1_score(val_labels, val_predictions_binary)
print("Validation Precision:", precision)
print("Validation Recall:", recall)
print("Validation F1-score:", f1)

# Plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(['Training', 'Validation'], [history.history['accuracy'][-1], 
history.history['val_accuracy'][-1]], label='Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.ylim(0, 1) # Limit y-axis to range [0, 1] for accuracy
plt.legend()

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.bar(['Training', 'Validation'], [history.history['loss'][-1], history.history['val_loss'][-
1]], label='Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()61
#testing the model

# Directory paths for test data
test_real_dir = r"E:\extractedFeatures\test\real"
test_fake_dir = r"E:\extractedFeatures\test\fake"

# Count the number of samples
test_real_samples = len([name for name in os.listdir(test_real_dir) if 
os.path.isfile(os.path.join(test_real_dir, name))])
test_fake_samples = len([name for name in os.listdir(test_fake_dir) if 
os.path.isfile(os.path.join(test_fake_dir, name))])

# Load test data generator
test_data_generator = data_generator(test_real_dir, test_fake_dir, batch_size)

# Evaluate the model on test data
test_steps = (test_real_samples + test_fake_samples) // batch_size
test_loss, test_accuracy = model.evaluate(test_data_generator, steps=test_steps)
print("Test Accuracy:", test_accuracy)
print("Test Loss:", test_loss)
#Prediction on video and black marking 
import numpy as np
from tensorflow.keras.models import load_model
import cv2
from mtcnn import MTCNN
import os

# Load the trained model
model = load_model(r"E:\LSTM_Model.h5")

# Function to predict labels for frames
def predict_labels(frames_features):
 predictions = model.predict(frames_features)
 return (predictions > 0.5).astype(int)
 
# Function to annotate the frame with a black mark for fake frames
def annotate_frame(frame, faces, is_fake):
 for face in faces:
  x, y, width, height = face['box']
if is_fake:
 cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 0), thickness=-1) # 
Fill face region with black
 return frame
 
# Path to directory containing pre-extracted features
features_dir = r"E:\extractedFeatures\test\fake\high quality\fadg0\sa1-video-fram1"

# Initialize MTCNN
detector = MTCNN()

# Load the frames
frames_dir = r"E:\splitted video dataset2\test\fake\high quality\fadg0\sa1-videofram1"
total_frames = 0
correct_predictions = 0
for frame_file in os.listdir(frames_dir):
 frame_path = os.path.join(frames_dir, frame_file)
 frame = cv2.imread(frame_path)
 if frame is None:
 continue
 
 # Load pre-extracted features
 feature_file = os.path.join(features_dir, frame_file.replace('.jpg', '_features.npy'))
 if not os.path.exists(feature_file):
 continue63
 features = np.load(feature_file)
 num_timesteps = 7
 
 # Assuming channel dimension is at index 2
 features = features[:, :, 0, :]
 features = features[0, :num_timesteps, :] # Truncate or pad the sequence
 
 # Pad features if necessary
 if features.shape[0] < num_timesteps:
  pad_width = ((0, num_timesteps - features.shape[0]), (0, 0))
features = np.pad(features, pad_width, mode='constant', constant_values=0)
 is_fake = predict_labels(features.reshape(1, num_timesteps, -1))[0][0] == 1 # 
Check if the frame is predicted as fake

 # Get the video name
 video_name = os.path.basename(frames_dir)
 
 # Detect faces in the frame
 faces = detector.detect_faces(frame)
 
 # Annotate the frame with black over detected faces if it's predicted as fake
 annotated_frame = annotate_frame(frame.copy(), faces, is_fake)
 total_frames += 1
 if is_fake:
  print(f"Video: {video_name}, Frame: {frame_file}, Prediction: Fake")
print("Fake Frame Detected!")
 else:
  print(f"Video: {video_name}, Frame: {frame_file}, Prediction: Real")
print("Real Frame Detected!")

 # Count correct predictions
 if (is_fake and 'fake' in frames_dir) or (not is_fake and 'real' in frames_dir):64
 correct_predictions += 1
 cv2.imshow("Original Frame", frame) # Display the original frame
 cv2.imshow("Annotated Frame", annotated_frame) # Display the annotated frame
 cv2.waitKey(60)
 print("Total Frames:", total_frames)
 print("Correct Predictions:", correct_predictions)
accuracy = correct_predictions / total_frames
print("Accuracy:", accuracy)
cv2.destroyAllWindows()
