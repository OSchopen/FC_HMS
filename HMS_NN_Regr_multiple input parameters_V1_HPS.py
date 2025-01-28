#=====NN With multiple input parameters====

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from queue import Queue
import tkinter as tk
from tkinter import Canvas, filedialog
from tkinter.messagebox import showinfo
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import threading
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

import datetime
from tensorflow import keras

import itertools
from tkinter import simpledialog



# Abfrage, ob Cuda für Tensorflow verfügbar ist
import tensorflow as tf
cuda_tensorflow=tf.test.is_built_with_cuda()
print("Cuda für Tensorflow ist verfügbar: ",cuda_tensorflow)


# Wähle CPU oder GPU für das Training in Tensorflow
device_choice = input("Möchten Sie mit der CPU oder GPU trainieren? (cpu/gpu): ").strip().lower()

if device_choice == 'gpu':
    # Überprüfen, ob GPUs verfügbar sind und konfigurieren
    gpus = tf.config.list_physical_devices('GPU')
    print("Available GPUs:", gpus)
    if gpus:
        try:
            # Setze das Speicherwachstum auf True
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # Wähle die erste GPU aus
            tf.config.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')   
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            
            #tf.config.set_logical_device_configuration(gpus[0],[tf.config.LogicalDeviceConfiguration(memory_limit=7000)])
            #config = tf.compat.v1.ConfigProto()
            #config.gpu_options.per_process_gpu_memory_fraction = 1
            #session = tf.compat.v1.InteractiveSession(config=config)
            
        except RuntimeError as e:
            # Speicherwachstum muss vor der Initialisierung des GPUs gesetzt werden
            print(e)
    else:
        print("Keine GPU gefunden. Verwenden der CPU stattdessen.")
        tf.config.set_visible_devices([], 'GPU')
else:
    # Verwenden Sie die CPU
    tf.config.set_visible_devices([], 'GPU')
    print("Training wird auf der CPU durchgeführt.")

# Zeigen Sie die aktuelle Device-Auswahl an
#tf.debugging.set_log_device_placement(False)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))



# Load Excel file .csv and read as Pandas DataFrame
def read_single_table():
    #Realpart 
    data_real = pd.read_csv('./HMS_Data base/Nyquist_Data_adapted_Real_part.csv', delimiter = ';', header = 0, decimal = '.')# read as pandas DataFrame 
    #print("data_real:\n",data_real)

    #Imaginarypart 
    data_imag = pd.read_csv('./HMS_Data base/Nyquist_Data_adapted_Imag_part.csv', delimiter = ';', header = 0, decimal = '.')# read as pandas DataFrame 
    #print(data_imag)

    return data_real, data_imag
[data_real, data_imag]=read_single_table()
print("data_real:\n",data_real)


# Load Excel file .xlsx and read as Pandas DataFrame
def read_multiple_table():
    file_path = './HMS_Data base/02_Nyquist_Data_extrapolation_new_adapted.xlsx'
    nyquist_data = pd.ExcelFile(file_path)

    # Load specific Excel-sheet 'Real' and 'Imag' in Excel file 
    df_real = nyquist_data.parse('Adapted', delimiter='\t', decimal=",")
    #df_imag = nyquist_data.parse('Imag', delimiter='\t', decimal=",")

    # Real part and imaginary part as Pandas Dataframe
    data_real=pd.DataFrame(df_real)
    print("Data_real:\n",data_real)

    return data_real
#data_real=read_multiple_table()


#extract Re and Im as a function of frequencie for one single operating point 
#define operating point as userinput 
#rel_air_humi = int(input("Enter relative Airhumidity (int values):"))
#Lambda = int(input("Enter Lambda value (int values):"))
#Temp = int(input("Enter Temperature (int values):"))
rel_air_humi=80
Lambda=2
Temp=55


# search for operating point in data
#Realpart
copy = data_real.where((data_real.RH_Air == rel_air_humi) )#& (data_real.Lambda == Lambda) )#& (data_real.Temperature == Temp))
#data_real_op_point = copy.dropna()
data_real_op_point = data_real.dropna()
print("\nData real op:\n",data_real_op_point)

#Imagpart 
#copy = data_imag.where((data_imag.RH_Air == rel_air_humi) & (data_imag.Lambda == Lambda) & (data_imag.Temperature == Temp))
#data_imag_op_point = copy.dropna()


# Assign X-Data set and y-Data

X=pd.DataFrame(data_real_op_point["Frequency"])
#X=pd.DataFrame(data_real_op_point.iloc[:,3])   -->Alternative
print("\nType of X-Data: ",type(X))
print("\nX-Data: \n"+str(X))


X_data_set_list=["RH_Air", "Lambda", "Temperature", "Frequency"]
X_data_set=pd.DataFrame(data_real_op_point[X_data_set_list])
#X_data_set=pd.DataFrame(data_real_op_point[{"RH_Air", "Lambda", "Temperature", "Frequency"}]) -->Alternative 1     
#X_data_set=pd.DataFrame(data_real_op_point.iloc[:,:4]) -->Alternative 2
print("\nType of X_data_set-Data: ",type(X_data_set))
print("\nX_data_set-Data: \n"+str(X_data_set))


y=data_real_op_point["Impedance_real_part"]
#y=data_real_op_point.iloc[:,4] --> Alternative
#y=y.transpose()
print("\nType of y-Data: ",type(y))
print("\ny-Data: \n"+str(y))


# Normalising X-Data
scaler = StandardScaler()
X=scaler.fit_transform(X)
print("Type of X-Data scaled: ",type(X))
print("X-Data shape: ", X.shape)
#print("X-Data scaled:\n", X)
X_den = pd.DataFrame(scaler.inverse_transform(X))


# Normalising X_data_set-Data
scaler = StandardScaler()
X_data_set = pd.DataFrame(scaler.fit_transform(X_data_set))
print("\nType of X_data_set-Data scaled: ",type(X_data_set))
print("\nX_data_set-Data shape: ", X_data_set.shape)
print("\nX_data_set-Data scaled:\n", X_data_set)
X_data_set_den=pd.DataFrame(scaler.inverse_transform(X_data_set))
#print("\nType of X_data_set-Data de-scaled: ",type(X_data_set_den))
#print("\nX-data_set de-scaled: \n",X_data_set_den)


# Seperation in training-, test, and validation-data
X_train_full, X_test, y_train_full, y_test = train_test_split(X_data_set, y, test_size = 0.5, random_state = 20, shuffle = True) # define radnom test, training and validation data 
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state = 20, shuffle = True)# from Training data full define validation data --> detect Overfit or underfit 

print("\nType of X_test: ",type(X_test))
print("\nX_test:\n",X_test)
print("\nType of y_test: ",type(y_test))
print("\ny_test:\n",y_test)

np.random.seed(42)
tf.random.set_seed(42)

# Print data shape
print("\nX_data_set-Data shape: ", X_data_set.shape)
print("X-Data shape:", X.shape)
print("X_data_set-Data shape[1:]: ", X_data_set.shape[1:],"\n")


# === Neural Network with hyperparameter study ===
global_epochs = []
global_batch_size=[]
global_losses = []

loss_table=pd.DataFrame()


def train_model_with_params(epochs, neurons, hidden_layers, activation, batch_size, learning_rate):
    global global_epochs, global_losses, global_RMSE_Train, global_RMSE_Test, model, loss_table

    global_epochs.clear()
    global_losses.clear()
    #global_RMSE_Train.clear()
    #global_RMSE_Test.clear()
    
    learning_rate=learning_rate/1000

    if activation == 'gelu':
        used_activation = tf.nn.gelu
    else:
        used_activation = activation

    model = Sequential()
    model.add(Dense(neurons, activation=used_activation, input_shape=X_data_set.shape[1:])), #input shape is: (4,))
    
    for _ in range(hidden_layers-1):
        model.add(Dense(neurons, activation=used_activation))
    model.add(Dense(1))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')

    epochs_count = epochs
    update_frequency = 1


    for epoch in range(epochs_count):
        print(f"Epoch {epoch + 1}/{epochs}")
        history = model.fit(X_train, y_train, epochs=1, batch_size=batch_size,  validation_data = (X_valid, y_valid)) # verbose=0 (Schaltet Fortschrittsanzeige aus)

        que_loss_table=pd.DataFrame(history.history)
        que_loss_table["MSE"]=model.evaluate(X_test, y_test)             # X-Data set oder X_test?
        que_loss_table["R²"]=r2_score(y_test, model.predict(X_test))    # y_test?     
        que_loss_table["Epoch"]=(f"{epoch + 1} / {epochs}")
        que_loss_table["Epochs"]=epochs
        que_loss_table["Activation"]=activation
        que_loss_table["Neurons"]=neurons
        que_loss_table["Hidden Layers"]=hidden_layers
        que_loss_table["Batch Size"]=batch_size
        que_loss_table["Learning Rate"]=learning_rate
        
        loss_table=loss_table.append(que_loss_table, ignore_index=True)

        # Ergebnisse einzeln kalkuliert
        #mse_test = model.evaluate(X_data_set, y)
        #y_train_pred = model.predict(X_train)
        #y_test_pred = model.predict(X_test)
        #train_r2 = r2_score(y_train, y_train_pred)
        #test_r2 = r2_score(y_test, y_test_pred)

        #global_epochs.append(epoch + 1)
        #global_losses.append(loss)
        #global_RMSE_Train.append(train_r2)
        #global_RMSE_Test.append(test_r2)

        #training_data_queue.put((X, y, X_train, y_train, X_test, y_test, X_den, X_train_den, X_test_den))

        #if epoch % update_frequency == 0:
            #window.after(0, lambda: update_plots())

        # Ausgabe des Fortschritts in der Konsole
        print("\nLoss_table:\n", loss_table,"\n")

        

    filename = f"./HMS_Parameter_study/mein_modell_ep{epochs}_neu{neurons}_lay{hidden_layers}_act{activation}_bat{batch_size}_lr{learning_rate}.h5"
    model.save(filename)
    print(f"Training mit aktuellem Setting abgeschlossen und Modell gespeichert unter: {filename}")
    
    filename_HPS=f'./HMS_Parameter_study/Parameter_study.csv'
    loss_table.to_csv(filename_HPS, index=False)
    print(f"Parameter Studie  abgeschlossen und Table gespeichert unter: {filename_HPS}\n")  

    return model, history, loss_table

config_count_label = None


def open_hyperparameter_dialog():
    dialog = tk.Toplevel(window)
    dialog.title("Hyperparameter Settings")

    tk.Label(dialog, text="Epochs (Start,End,Step):").grid(row=0, column=0, sticky='w')
    epochs_entry = tk.Entry(dialog)
    epochs_entry.insert(0, "100,100,100")
    epochs_entry.grid(row=0, column=1)

    tk.Label(dialog, text="Neurons (Start,End,Step):").grid(row=1, column=0, sticky='w')
    neurons_entry = tk.Entry(dialog)
    neurons_entry.insert(0, "5,5,5")
    neurons_entry.grid(row=1, column=1)

    tk.Label(dialog, text="Hidden Layers (Start,End,Step):").grid(row=2, column=0, sticky='w')
    layers_entry = tk.Entry(dialog)
    layers_entry.insert(0, "1,1,1")
    layers_entry.grid(row=2, column=1)

    tk.Label(dialog, text="Batch Size (Start,End,Step):").grid(row=3, column=0, sticky='w')
    batch_size_entry = tk.Entry(dialog)
    batch_size_entry.insert(0, "32,32,32")
    batch_size_entry.grid(row=3, column=1)

    tk.Label(dialog, text="Learnig Rate * 10^-3 (Start,End,Step):").grid(row=4, column=0, sticky='w')
    learning_rate_entry = tk.Entry(dialog)
    learning_rate_entry.insert(0, "1,1,1")
    learning_rate_entry.grid(row=4, column=1)
    
    tk.Label(dialog, text="Activation Functions:").grid(row=5, column=0, sticky='w')
    relu_var = tk.BooleanVar()
    gelu_var = tk.BooleanVar()
    tanh_var = tk.BooleanVar()
    elu_var = tk.BooleanVar()
    leaky_relu_var=tk.BooleanVar()
    sigmoid_var=tk.BooleanVar()
    softmax_var=tk.BooleanVar()
    softplus_var=tk.BooleanVar()

    cb_relu = tk.Checkbutton(dialog, text="relu", variable=relu_var)
    cb_relu.grid(row=5, column=1, sticky='w')
    cb_gelu = tk.Checkbutton(dialog, text="gelu", variable=gelu_var)
    cb_gelu.grid(row=6, column=1, sticky='w')
    cb_tanh = tk.Checkbutton(dialog, text="tanh", variable=tanh_var)
    cb_tanh.grid(row=7, column=1, sticky='w')

    cb_elu = tk.Checkbutton(dialog, text="elu", variable=elu_var)
    cb_elu.grid(row=8, column=1, sticky='w')
    cb_leaky_relu = tk.Checkbutton(dialog, text="leaky_relu", variable=leaky_relu_var)
    cb_leaky_relu.grid(row=5, column=2, sticky='w')
    cb_sigmoid = tk.Checkbutton(dialog, text="sigmoid", variable=sigmoid_var)
    cb_sigmoid.grid(row=6, column=2, sticky='w')
    cb_softmax = tk.Checkbutton(dialog, text="softmax", variable=softmax_var)
    cb_softmax.grid(row=7, column=2, sticky='w')
    cb_softplus = tk.Checkbutton(dialog, text="softplus", variable=softplus_var)
    cb_softplus.grid(row=8, column=2, sticky='w')

    global config_count_label
    config_count_label = tk.Label(dialog, text="Total Configurations: 0")
    config_count_label.grid(row=9, column=0, columnspan=2, pady=5)

    def parse_range(range_str):
        start, end, step = range_str.split(',')
        return range(int(start), int(end)+1, int(step))

    def update_config_count():
        ep_range = parse_range(epochs_entry.get())
        neu_range = parse_range(neurons_entry.get())
        lay_range = parse_range(layers_entry.get())
        bat_range = parse_range(batch_size_entry.get())
        lr_range = parse_range(learning_rate_entry.get())

        

        activations_selected = []
        if relu_var.get():
            activations_selected.append("relu")
        if gelu_var.get():
            activations_selected.append("gelu")
        if tanh_var.get():
            activations_selected.append("tanh")
        if elu_var.get():
            activations_selected.append("elu")
        if leaky_relu_var.get():
            activations_selected.append("leaky_relu")
        if sigmoid_var.get():
            activations_selected.append("sigmoid")
        if softmax_var.get():
            activations_selected.append("softmax")
        if softplus_var.get():
            activations_selected.append("softplus")

        count = (len(ep_range) * len(neu_range) * len(lay_range) * len(bat_range)*len(activations_selected)*len(lr_range))
        config_count_label.config(text=f"Total Configurations: {count}")

    def start_study():
        activations_selected = []
        if relu_var.get():
            activations_selected.append("relu")
        if gelu_var.get():
            activations_selected.append("gelu")
        if tanh_var.get():
            activations_selected.append("tanh")
        if elu_var.get():
            activations_selected.append("elu")
        if leaky_relu_var.get():
            activations_selected.append("leaky_relu")
        if sigmoid_var.get():
            activations_selected.append("sigmoid")
        if softmax_var.get():
            activations_selected.append("softmax")
        if softplus_var.get():
            activations_selected.append("softplus")

        if len(activations_selected) == 0:
            showinfo("Fehler", "Bitte mindestens eine Aktivierungsfunktion auswählen.")
            return

        # Werte zuerst auslesen, bevor Dialog zerstört wird
        ep = epochs_entry.get()
        neu = neurons_entry.get()
        lay = layers_entry.get()
        bat= batch_size_entry.get()
        lr=learning_rate_entry.get()

        dialog.destroy()
        start_hyperparameter_study(ep, neu, lay, activations_selected, bat, lr)

    update_button = tk.Button(dialog, text="Update Config Count", command=update_config_count)
    update_button.grid(row=10, column=0, pady=10)

    start_button = tk.Button(dialog, text="Start Hyperparameter Study", command=start_study)
    start_button.grid(row=10, column=1, pady=10)



def start_hyperparameter_study(epochs_str, neurons_str, layers_str, activations_selected, batch_str, learning_rate_str):
    def parse_range(range_str):
        start, end, step = range_str.split(',')
        return range(int(start), int(end)+1, int(step))

    epochs_range = parse_range(epochs_str)
    neurons_range = parse_range(neurons_str)
    layers_range = parse_range(layers_str)
    batch_size_range=parse_range(batch_str)
    learning_rate_range=parse_range(learning_rate_str) 

    for (ep, neu, lay, act, bat, lr) in itertools.product(epochs_range, neurons_range, layers_range, activations_selected, batch_size_range, learning_rate_range):
        print(f"Starte Training mit Epochen={ep}, Neuronen={neu}, Hidden Layers={lay}, Activation={act}, Batch Size={bat}, Learning rate={lr/1000}")
        train_model_with_params(epochs=ep, neurons=neu, hidden_layers=lay, activation=act, batch_size=bat, learning_rate=lr)



window = tk.Tk()
open_hyperparameter_dialog()
window.mainloop()
# === Neural Network with hyperparameter study end ===






