import tkinter as tk  # Importing the tkinter library for GUI
from tkinter import filedialog  # Importing filedialog submodule for file selection
from PIL import Image, ImageTk  # Importing Image and ImageTk from PIL for image processing
import numpy as np  # Importing numpy for numerical operations
from keras.models import load_model  # Importing load_model function from Keras for loading pre-trained model

# Load the trained model
model = load_model("fruit.h5")

# Define the fruit classes
classes = ['Apple Braeburn', 'Apple Granny Smith', 'Apricot', 'Avocado', 'Banana', 'Blueberry',
           'Cactus fruit', 'Cantaloupe', 'Cherry', 'Clementine', 'Corn', 'Cucumber Ripe',
           'Grape Blue', 'Kiwi', 'Lemon', 'Limes', 'Mango', 'Onion White', 'Orange',
           'Papaya', 'Passion Fruit', 'Peach', 'Pear', 'Pepper Green', 'Pepper Red',
           'Pineapple', 'Plum', 'Pomegranate', 'Potato Red', 'Raspberry', 'Strawberry',
           'Tomato', 'Watermelon']

# Function to preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path)  # Open the image file
    img = img.resize((100, 100))  # Resize the image to match the input size of the model
    img = np.array(img) / 255.0  # Normalize the pixel values to [0,1]
    return img

# Function to classify the image
def classify_image(image_path):
    img = preprocess_image(image_path)  # Preprocess the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    pred = model.predict(img)  # Perform prediction
    class_idx = np.argmax(pred)  # Get the index of the predicted class
    return classes[class_idx]  # Return the class label corresponding to the index

# Function to handle the "Browse" button click event
def browse_file():
    file_path = filedialog.askopenfilename()  # Open file dialog to select an image
    if file_path:  # If a file is selected
        result_label.config(text="Classifying...")  # Update result label to show classifying message
        result = classify_image(file_path)  # Classify the selected image
        result_label.config(text="Predicted Class: " + result)  # Update result label with predicted class

# Create the main window
root = tk.Tk()  # Create an instance of Tk class
root.title("Fruit Classifier")  # Set the title of the window

# Set window dimensions
window_width = 550
window_height = 650
root.geometry(f"{window_width}x{window_height}")  # Set the dimensions of the window

# Load and resize the background image
bg_image = Image.open("background_image.jpg")  # Open the background image
bg_image = bg_image.resize((window_width, window_height), Image.ANTIALIAS)  # Resize the background image
bg_photo = ImageTk.PhotoImage(bg_image)  # Convert the image to PhotoImage format
bg_label = tk.Label(root, image=bg_photo)  # Create a label to display the background image
bg_label.place(x=0, y=0, relwidth=1, relheight=1)  # Place the label at (0,0) with full width and height

# Create a button for browsing images
browse_button = tk.Button(root, text="Browse", command=browse_file, font=('Comic Sans MS', 20), fg="#F800F8", bg="black")
browse_button.place(x=50, y=350)  # Place the button at (50,350)

# Create a label for displaying the result
result_label = tk.Label(root, text="", font=('Comic Sans MS', 18), fg="#F800F8", bg="black")
result_label.place(x=50, y=450)  # Place the label at (50,450)

root.mainloop()  # Start the GUI event loop
