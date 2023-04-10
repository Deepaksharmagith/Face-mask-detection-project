from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

#Define the Model
mymodel=Sequential()
mymodel.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
mymodel.add(MaxPooling2D())
mymodel.add(Conv2D(32,(3,3),activation='relu'))
mymodel.add(MaxPooling2D())
mymodel.add(Conv2D(32,(3,3),activation='relu'))
mymodel.add(MaxPooling2D())
mymodel.add(Flatten())
mymodel.add(Dense(1,activation='sigmoid'))
mymodel.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])

#Organize the Data
train=ImageDataGenerator(rescale=1./255)
test=ImageDataGenerator(rescale=1./255)
train_set=train.flow_from_directory('train',target_size=(150,150),batch_size=16,class_mode='binary')
test_set=test.flow_from_directory('test',target_size=(150,150),batch_size=16,class_mode='binary')

#Train the model
k=mymodel.fit(train_set,epochs=10,validation_data=test_set)

#Save the model
mymodel.save('mask.h5',k)
