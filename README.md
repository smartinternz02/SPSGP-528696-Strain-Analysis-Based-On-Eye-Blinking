# Strain-Analysis-Based-On-Eye-Blinking    

Blinking is a reflex, which means your body does it automatically. Babies and children only blink about two times per minute. By the time you reach adolescence that increases to 14 to 17 times per minute. 
Detecting eye blinks is important for instance in systems that monitor a human operator vigilance, e.g. driver drowsiness, in systems that warn a computer user staring at the screen without blinking for a long time to prevent the dry eye and the computer vision syndromes, in human-computer interfaces that ease communication for disabled people. There should be an application that monitors to let the user know that he might get strained.

## Demo Video
https://drive.google.com/file/d/1b_hrTwjrwaee92QzqEWEJ2rkCG6SlvQx/view?usp=sharing

## Import Necessary Libraries
The first step is usually importing the libraries that will be needed in the program.
The required libraries to be imported to  Python script are:

#### SciPy:

It is a scientific library for Python is an open-source, a licensed library for mathematics, science, and engineering. The SciPy library depends on NumPy and provides many user-friendly and efficient numerical practices such as routines for numerical integration, optimization, distance calculation.

#### Imutils:

Imutils package has  a series of convenience functions to make basic image processing functions such as translation, rotation, resizing, and displaying Matplotlib images and video frames easier with OpenCV
Here we are importing File VideoStream and VideoStream from the imutils package.

#### Numpy:

It is an open-source numerical Python library. It contains a multi-dimensional array and matrix data structures. It can be used to perform mathematical operations on arrays such as trigonometric, statistical, and algebraic routines.

#### argparse:

The argparse module is used to write user-friendly command-line interfaces. The program defines the required arguments and the argparse will figure out how to parse those out of sys.argv.
This module automatically generates help and usage messages and issues errors when users give the program invalid arguments. 

#### time:

time module provides various time-related functions such as datetime , calendar, etc.,

#### datetime:

datetime function supplies classes for manipulating dates and times.

#### dlib:

dlib is an open-source library used for face detection. We recommend installing dlib updated version for better compatibility with OpenCV. Given a face, dlib which is a pre-trained model can extract features from the face like eyes, nose, lips, and jaw using facial landmarks. 

#### OpenCV:

OpenCV is a library of programming functions mainly aimed at real-time computer vision. Here, OpenCV is used to capture frames by accessing the webcam in real-time.

#### Google text to speech 

gTTS (Google Text-to-Speech), a Python library, and CLI tool to interface with Google Translate’s text-to-speech API.
Writes spoken mp3 data to a file, a file-like object (bytestring) for further audio manipulation, or stdout. It features flexible pre-processing and tokenizing, as well as automatic retrieval of supported languages.

#### playsound

The playground module is a cross-platform module that can play audio files. This doesn’t have any dependencies. We are using the play sound module in the project to alert the user about the number of blinks.

#### Tkinter

Tkinter is a graphical user interface (GUI) module for Python, you can make desktop apps with Python. You can make windows, buttons, show text, and images amongst other things.
In our project, we are using Tkinter for popup message.
All the above modules can be imported into our program using the below code
## Dependencies
You also need shape detector, you can download it by
https://www.kaggle.com/datasets/codebreaker619/face-landmark-shape-predictor?resource=download


## Command to Run:
python app.py --shape-predictor shape_predictor_68_face_landmarks.dat
