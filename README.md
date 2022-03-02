# facial-recognition-vgg16

detect and recognise multi faces in camera or webcam with trained model (vgg16) and save in history

## DataCreator 

### with this code you can create dataset with your own face or other faces from webcam and images.

-- webcam: it can collect n images from webcam and set destination to save them (i recommand atleast capture 50 images)

-- storage : with this function you can set source path of your images and set destination path to detect faces in image and save them

## create model

### in this code i craete vgg16 model and three layes at top that you can create model and save it from your dateset.

-- you can change layers and number of classes and set path of dataset

## facial recogniton

### in this code you can use youe created model and detect and recognize multi faces and save them in history file with format '*.csv' .

-- csv contain={Name, StartTime, EndTime}

---- Name = name of person that decteded

---- StartTime = time that person detected

---- EndTime = time that person can't detect anymore
