import os
import numpy as np

def split_train_test(filenames, train_split = 0.8, validation_split = 0.1, test_split = 0.1):
    # creating folder called "Train" and "Test" under "data" folder


    if os.path.isdir(os.path.join("..", "data", "train")) == False:
        os.mkdir(os.path.join("..", "data", "train"))
    if os.path.isdir(os.path.join("..", "data", "test")) == False:
        os.mkdir(os.path.join("..", "data", "test"))
    if os.path.isdir(os.path.join("..", "data", "validation")) == False:
        os.mkdir(os.path.join("..", "data", "validation"))

    # Creating "Darth Vader", "Stormtrooper", and "Yoda" folders in train and test folders
    for name in ["Darth Vader", "Stormtrooper", "Yoda"]:
        if os.path.isdir(os.path.join("..", "data", "train", name)) == False:
            os.mkdir(os.path.join("..", "data", "train", name))
        if os.path.isdir(os.path.join("..", "data", "test", name)) == False:
            os.mkdir(os.path.join("..", "data", "test", name))
        if os.path.isdir(os.path.join("..", "data", "validation", name)) == False:
            os.mkdir(os.path.join("..", "data", "validation", name))

    for idx, name in enumerate(["Darth Vader", "Stormtrooper", "Yoda"]):
        np.random.shuffle(filenames[name])

        train_size = np.int(train_split*len(filenames[name]))
        validation_size = np.int(validation_split*len(filenames[name]))
        test_size = len(filenames[name]) - train_size - validation_size

        train_files  = filenames[name][:train_size]
        validation_files = filenames[name][train_size:train_size+validation_size]
        test_files = filenames[name][train_size+validation_size:train_size+validation_size+test_size]
        
        for train in train_files:
            img = train.split(os.sep)[-1]
            try:
                os.rename(train, os.path.join("..", "data", "train", name, img))
            except:
                pass
        for test in test_files:
            img = test.split(os.sep)[-1]
            try:
                os.rename(test, os.path.join("..", "data", "test", name,img))
            except:
                pass
        for validation in validation_files:
            img = validation.split(os.sep)[-1]
            try:
                os.rename(validation, os.path.join("..", "data", "validation", name,img))
            except:
                pass
        os.rmdir(os.path.join("..", "data", name))
        
    return
