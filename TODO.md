# TODO
Agregar ejemplos de torchvision para lee una imagen y marcar los bounding boxes. Hacer un ejemplo que lea la imagen, la
transforme en tensor y la pase al dibujo. Poner cabeceras a todos los archivos Completar el ciclo de entrenamiento con
el archivo .pth Pasar al nuevo Github con fotos.

## Ejemplos de conversiones entre vectores
ndarray to Tensor
Image to Tensor
Resolver el warning
C:\Users\Usuari\development_env\object_detector_venv\lib\site-packages\torchvision\models\_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13
https://www.geeksforgeeks.org/how-to-read-a-jpeg-or-png-image-in-pytorch/




## Common operations made with Tensors
https://github.com/Tgaaly/pytorch-cheatsheet
https://kunpengsong.github.io/2019/06/Pytorch-cheat-sheet/
https://stackoverflow.com/questions/43327668/looping-over-a-tensor

## PENDING TASKS
1) output with merged masks in binary [OK] test_mask_02.py
2) output with colour image and binary mask overlapped [OK] test_mask_02.py
3) output with bounding box + mask detected + coloured image [OK] test_mask_03.py
4) process results from model with bounding box + mask detected + coloured image mask + binary mask. [OK] test_mask_04.py
5) Process results from model and merge the mask
6) Process all the images from a folder and merge the mask.
7) bounding boxes
8) Filter detections by threshold and label
9) Create bounding box with label
Filter by label name.
Colour fixed, random colours
Crete own dataset with folders
Organise routines.
Train with own labels, modify type of labels
Test inference time
Load saved and trained model.
Save detected bounding boxes to files.

## Training
### Data augmentation

### Dataset management

### GPU settings in Windows 10
Use of GPU device in Windows 10. [OK]Solved, it is needed to install CUDA 11.7 and then.
Train in Linux
Train in Windows.
Create own dataset.


# Tensor conversion
https://discuss.pytorch.org/t/how-to-use-pytorch-tensor-object-in-opencv-without-convert-to-numpy-array/66000/7
https://discuss.pytorch.org/t/pytorch-pil-to-tensor-and-vice-versa/6312
https://www.google.com/search?client=firefox-b-d&q=ndarray+to+tensor+pytorch
https://pydlt.readthedocs.io/en/latest/source/quickstart/images.html
https://www.tutorialspoint.com/how-to-convert-an-image-to-a-pytorch-tensor
https://stackoverflow.com/questions/66237451/convert-numpy-ndarray-to-pil-and-and-convert-it-to-tensor
https://www.geeksforgeeks.org/converting-an-image-to-a-torch-tensor-in-python/


Images are 1025 * 205 in .png format 
Add links in README.md to examples.
