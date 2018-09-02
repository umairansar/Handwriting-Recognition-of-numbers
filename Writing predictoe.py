from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels
import matplotlib.pyplot as plt
import numpy as np

# Extracting Files
with open('C:\\Users\\sunlight\\.spyder-py3\\train-images-idx3-ubyte.gz', 'rb') as f:
    train_images = extract_images(f)
with open('C:\\Users\\sunlight\\.spyder-py3\\train-labels-idx1-ubyte.gz', 'rb') as f:
    train_labels = extract_labels(f)
with open('C:\\Users\\sunlight\\.spyder-py3\\t10k-images-idx3-ubyte.gz', 'rb') as f:
    test_images = extract_images(f)
with open('C:\\Users\\sunlight\\.spyder-py3\\t10k-labels-idx1-ubyte.gz', 'rb') as f:
    test_labels = extract_labels(f)
        
    
    #because a single train_image shape is (28,28,1) here 1 means the image is greyscale. If 1 
    #is replaced by (3 or 4) then the image would be consiered RGB i.e. coloured image.
print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)
print(train_images[0].shape)


# Squese function is used to remove the unnecessary dimension to be able to print it on screen
#i.e. convert (28,28,1) to (28,28) so the invalid dimension error does'nt appear. Other way would have been
#that the images originally had either(28,28,3) or (28,28,4) to be successfully printed by pyplot.
random_img = train_images[0].squeeze()  
print(random_img.shape)
plt.imshow(random_img, cmap='gray')
plt.show()


# One-hot_encoding to encode trained_labels
from keras.utils import to_categorical
encoded = to_categorical(train_labels)
print(encoded[0])

#Flatten the images
flattenedTrainedImages=train_images.reshape(len(train_images),-1) # len(train_images) output: 60000
print(flattenedTrainedImages.shape) # This is arrau flattened image with 60000 images(rows) and 784 pixels (columns).
                                                                                            # 28 * 28 = 784
# Creating Neural Network
import tensorflow as tf                                                                                   
x = tf.placeholder(tf.float32, shape = [None, 784])
y = tf.placeholder(tf.float32, shape = [None, 10])  # 10 because we have 0 to 9 total possible input number

weights = tf.Variable(tf.random_normal([784,10]))
biases = tf.Variable(tf.random_normal([1,10]))


scores = tf.matmul(x,weights) + biases
pred = tf.nn.softmax(scores)        # to get probabiblity destribution
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = scores))
optimizer = tf.train.AdamOptimizer(0.5) #Adam is better than gradient descent
train = optimizer.minimize(cost)


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(pred,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

#batch_size = 128
with tf.Session() as sess:
    sess.run(init)
    for i in range(100):
        sess.run(train, feed_dict={x : flattenedTrainedImages, y : encoded })
        print(sess.run(cost,{x:flattenedTrainedImages, y:encoded}))

    print(sess.run(accuracy, feed_dict = {x:flattenedTrainedImages, y:encoded}))

    randomNumber = np.random.randint(0,10000) 
    image = test_images[randomNumber].reshape(1,784)
    display_image = image.reshape(28,28)
    plt.imshow(display_image, cmap='gray')
    plt.show()
    print('Predicted Number : ', np.argmax(sess.run(pred, feed_dict={x:image})))
    

    #When hidden layer is used gradient decent optimizer might explode the value, so use instead "AdamOptimizor"



# print(pred, image)   










