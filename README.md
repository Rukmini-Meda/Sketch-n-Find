# Sketch-n-Find
Our project Sketch-n-Find is a sketch based image retrieval system. Now, you can just sketch out the picture in your mind and our application will give you the best possible set of similar images.

 - Dataset 
   First large scale dataset of sketch - photo pairs
    125 Categories
    75, 471 sketches
    12, 500 photos
    256 x 256
  Dataset can obtained from this site [Sketchy dataset website](http://sketchy.eye.gatech.edu/)
  
  - Data Preprocessing
    data/dataloader.py is used for the Trainloader,Validation loader,Testloader
    
    data/datasets.py is used for get the positive and negative labels for the anchor images 
    
 - Sampling the Data
   After loading the data we subsampled the dataset so that we can have less number of images from each category
   
   create_mini_dataset.py code is used for subsampling the data
   
     - Run this file 2 times one with photo folder path and one with sketch folder path
     - Adjust the total variable for subsampling the data as needed
    
  - Models
    Used Triplet Network with triplet margin loss
    
    models/basenet.py is code for basic architecture
    
    models/Triplet.py is code for triplet network
    
 - Training
   train.py is code for training give hyperparameters
   
 - Testing
    test.py s code for testing 
   
