# PA_KB_B2_6_2020
import zipfile : untuk mengekstrak unzip
import tensorflow as tf : untuk library mengolah machine learning
import numpy as np : untuk operasi matematika yang lebih kompleks khusus nya untuk array/matrix
import splitfolders : untuk nge splitt folder dataset
import seaborn as sns : untuk memvisualisasikan data
from tensorflow.keras.preprocessing.image import ImageDataGenerator : untuk proses augmentasi setiap directory data akan di augmentasi menggunakan image data generator 
import matplotlib.pyplot as plt : untuk menampilkan visualisasi data
import matplotlib.image as mpimg : untuk menampilkan visualisasi gambar
import os : untuk utilitas sistem opersi (buat folder,lihat isi folder, dll)

#untuk mengambil dataset dari google drive.
from pydrive.auth import GoogleAuth 
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# authenticate and membuat drive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

file_id = '1Qa83l7exEqfHljP-DRyyO8sqcXFRGPQt'
downloaded = drive.CreateFile({'id':file_id})
downloaded.FetchMetadata(fetch_all=True)
downloaded.GetContentFile(downloaded.metadata['title'])

local_zip = 'chest_xray.zip'= zip dataset bernanama chest_xray.zip
zip_ref = zipfile.ZipFile(local_zip, 'r') = zip dari folder
zip_ref.extractall('') = mengekstrak semua file
zip_ref.close() = mengerluarkan file zip

# os.mkdir('ready_dataset')
base_dir = 'chest_xray' = lokasi directory paling awal atau yang paling besar
os.listdir(base_dir) = menampilkan directory keseluruhan

normal_dir = os.path.join(base_dir, 'NORMAL') = penglabelan/penamaan dari directory normal
pneumonia_dir = os.path.join(base_dir, 'PNEUMONIA') = penglabelan/penamaan dari directory pneumonia

class_name = ['NORMAL', 'PNEUMONIA'] / nama class dataset yang di gunakan

list_jumlah_file_all = [] 

# loop untuk menyimpan jumlah masing-masing file tiap kelas kedalam list
for kelas in os.listdir(base_dir):
    list_file_kelas = os.listdir(os.path.join(base_dir, kelas)) 
    banyaknya_file = len(list_file_kelas)
    list_jumlah_file_all.append(banyaknya_file) # masukkan jumlahnya ke dalam list
    
x = class_name # variabel yang berisi classname
y = list_jumlah_file_all # list dari banyaknya jumlah file keseluruhan yang diambil dari perulangan diatas
f = plt.figure() = visualisasi data normal dan pneumonia menggunakan diagram batang
f.set_figwidth(11) = lebar diagram batang
f.set_figheight(5) = tinggi diagram batang
plt.bar(x, y)
plt.title('Jumlah Dataset Keseluruhan', loc='center') = penamaan label di bagian atas diagram
plt.xlabel('PNEUMONIA & NORMAL') = penamaan label di bagian bawah diagram
plt.ylabel('Jumlah Data') = penamaan label di bagian kiri diagram
plt.show()= menampilkan diagram

splitfolders.ratio(base_dir, output = 'ready_dataset', seed = 1337, ratio = (.7,0.1,0.2)) = untuk mengsplit data menjadi 3 folder(trainng, validasi , testing)

train_dir = os.path.join('ready_dataset', 'train') : menggabungkan jalur folder
validation_dir = os.path.join('ready_dataset', 'val') :  menggabungkan jalur folder
test_dir = os.path.join('ready_dataset', 'test') :  menggabungkan jalur folder
os.listdir(train_dir) : menampilkan directory isi data train
os.listdir(validation_dir) : menampilkan directory isi data validation
os.listdir(test_dir) : menampilkan directory isi data test
# normal_dir = os.path.join(base_dir, 'NORMAL')
# pneumonia_dir = os.path.join(base_dir, 'PNEUMONIA')

Data Augmentation :
def to_grayscale(image): membuat fungsi untuk greyscale image
    return tf.image.rgb_to_grayscale(image) = mengembalikan gambar yang sudah di grayscale

posisi augmentasi :
train_datagen = ImageDataGenerator(
    rescale = 1./255, : merubah skala gambar
    rotation_range=20, : merotasi gambar
    horizontal_flip=True, : membalik secara vertikal
    shear_range=0.2, : untuk mengatur skala image
    zoom_range=0.2,: value < 1 akan zoom in
    height_shift_range=0.2, : untuk mengatur tinggi skala
    width_shift_range=0.2, : untuk mengatur lebar skala
    fill_mode='nearest') : # untuk mengisi gambar atau wadah yang tidak memiliki nilai
                    #preprocessing_function=to_grayscale_then_rgb,

test_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

color augmentasi :
train_generator = train_datagen.flow_from_directory(
  train_dir, : untuk direktori data train
  target_size=(150,150), : mengubah resolusi seluruh gambar jadi 150,150
  color_mode='grayscale', : merubah warna image dari grayscale
  class_mode='binary',  : karena hanya 2 clasification menggunakan binary
  shuffle=True, : unuk mengacak gambar terus-menerus
  batch_size=64) : buat menentukan step per epochs

validation_generator = validation_datagen.flow_from_directory(
  validation_dir,  
  target_size=(150,150,1),: mengubah resolusi seluruh gambar jadi 150,150
  color_mode='grayscale', : merubah warna image dari grayscale
  class_mode='binary',  : karena hanya 2 clasification menggunakan binary
  shuffle=True, : unuk mengacak gambar terus-menerus
  batch_size=64): buat menentukan step per epochs

test_generator = test_datagen.flow_from_directory(
  test_dir,  
  target_size=(150,150),
  color_mode='grayscale',
  class_mode='binary',
  shuffle=True, : unuk mengacak gambar terus-menerus
  batch_size=64): buat menentukan step per epochs

.....

menampilkan data agmentasi beserta labae :
plt.figure(figsize=(7,7)) : menentukan ukuran canvas
for a, i in enumerate(range(10,19)): memecah untuk di masukkan ke a dan i
    ax = plt.subplot(3,3,a+1) : dimensi 3 kali 3 a+1 itu index
    plt.imshow(image[i].reshape((150,150))) : menampilkan gambar
    plt.title(class_name[int(label[i])]) : memberikan nama
    plt.axis("off") :

Modelling :

Neuron/layer :
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), strides=1, padding='same', activation='relu', input_shape=(150,150,1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D((2,2), strides=2, padding='same'),
    tf.keras.layers.Conv2D(64, (3,3), strides=1, padding='same', activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D((2,2) , strides = 2 , padding = 'same'),
    tf.keras.layers.Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D((2,2) , strides = 2 , padding = 'same'),
    tf.keras.layers.Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D((2,2) , strides = 2 , padding = 'same'),
    tf.keras.layers.Conv2D(256 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(), :
    tf.keras.layers.MaxPool2D((2,2) , strides = 2 , padding = 'same'),
    tf.keras.layers.Flatten(), : menyatukan input yang memiliki dimensi
    tf.keras.layers.Dense(units = 128 , activation = 'relu'),
    tf.keras.layers.Dropout(0.2), : biar tidak jadi overfitting
    tf.keras.layers.Dense(units = 1 , activation = 'sigmoid')

...
menampilkan jumlah history atau akurasi data :
history = model.fit(train_generator,
                    epochs=100,
                    validation_data=validation_generator,
                    callbacks=[reduce_lr, early_stopping])


epochs = [i for i in range(21)] = menghasilkan list dari 0-21
fig , ax = plt.subplots(1,2) = untuk membuat satu plot memiliki 1 baris 2kolom
train_acc = history.history['accuracy'] =  menalmpilkan history akurasi
train_loss = history.history['loss'] =  menalmpilkan history akurasi
val_acc = history.history['val_accuracy'] = menalmpilkan history akurasi validasi
val_loss = history.history['val_loss']  = menalmpilkan history loss validasi
fig.set_size_inches(20,10)

ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
ax[0].plot(epochs , val_acc , 'ro-' , label = 'Validation Accuracy')
ax[0].set_title('Training & Validation Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
ax[1].plot(epochs , val_loss , 'r-o' , label = 'Validation Loss')
ax[1].set_title('Testing Accuracy & Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs") = penamaan label di bagian bawah grafik
ax[1].set_ylabel("Training & Validation Loss") = penamaan label pada bagian atas grafik
plt.show() = menampilkan keseluruhan grafik

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data") = mencetak tulisan Evaluate on test data
results = model.evaluate(test_generator) = mengevaluasi data melalui test generator
print("test loss:", results[0]) =  mencetak hasil test loss
print("test acc:", results[1]) = mencetaj hasil test akurasi

plt.figure(figsize=(15,10)) = ukuran diagram 15,10
for i in range(9) : =  memiliki range 9
    TrueLabel = class_name[int(label[i])] = penamaan pada gambar
    plt.subplot(3,3,i+1) = posisi dari gambar
    plt.axis('off') = menghilangkan sumbu x, y 
    y_pred = int(model.predict(image[i].reshape((1,150,150,1)),verbose=0)) = untuk prediksi dari gambar
    plt.imshow(tf.squeeze(image[i])) =  menampilkan gambar 
    plt.title(f'label: {TrueLabel}, predict : {class_name[y_pred]}') = menampilkan gambar apakah dia normal atau pneumonia