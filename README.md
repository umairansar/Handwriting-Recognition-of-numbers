# Handwriting-Recognition-of-numbers
Using the famous MNIST dataset, I have build a model with accuracy as good as 92.09%.

### Methodology:
The model was built using the Google's Tensorflow framework. The approach can be broken down as follows:
        
        1- Extraction of images and corresponding labels.
        2- Preprocessing of image data, such as flattening and converting to greyscale.
        3- Preprocessing of label data to encode them using 'One Hot Enciding.'
        4- Setting placeholders and function approximator variables.
        5- Running a session to train the data.
        6- Testing accuracy on a single image by getting prediction by model.
        
I have uploaded the code with this repository and you will see the same flow there as described above.

### Challenges:

1) During the building, there were a number of decisions to be made.  Since there are a number of laber encoders availible, you need to select one that suits to your requirement. Label Encoder, One Hot Encoding, Local Binarizer are all kinds of encoder. 
Label Encoder allots integer values to labels and that might interfere with the significance of a label as a label with larger nunber encoded might become more significant in the eyes of the model; thus, biasing the predicion. Local binarizer is used to encode string labels.

2) Using Gradient Descent optimzer might explode the cost values(i.e. NaN). This can also be attributed to the Adam Optimzer ability to vary (reduce) learning rate as the epoch number increases.


### Resolution and Soltion:
1) I used One Hot Encoding since it uses a zeros and ones matrix with a single one in each matrix. That one corresponds to the class i.e. its position in the matrix.

2) Use Adam Optimizer. This can also be attributed to the Adam Optimzer ability to vary (reduce) learning rate as the epoch number increases.

### Final Outcome:
The model was successfully trained on 50k images from MNIST dataset, returining an accuracy of 92.09%

### Future Work:
This single object image classification can be revamped to recognize multiple numbers within an image, and this technique is already being used in text recognition softwares.

#### Dataset Link : http://yann.lecun.com/exdb/mnist/


