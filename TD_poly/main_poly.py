from keras import optimizers
from keras import losses

import tensorflow as tf
tf.keras.backend.set_floatx('float64')
import os
#os.chdir("/home/bbensaid/Documents/Anabase/NN_shaman") 

import activations_perso
from model import build_poly
from training import train
import read
import numpy as np
import matplotlib.pyplot as plt

batch_size=2
x_train,y_train = read.poly_data()
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=batch_size).batch(batch_size)

# polyThree
activation=activations_perso.polyThree
loss = losses.MeanSquaredError()
name_init="Uniform"
w=-2; b=2
params_init=[w,w,b,b]
seed=0

#paramètres d'arrêt
eps=10**(-6); max_epochs=30000

#paramètres d'entrainement 
lr=0.1
seuil=0.01
f1=30; f2=10000; lambd=0.5; rho=0.9; eps_egd=0.01
beta_1=0.9; beta_2=0.999; epsilon=1e-07
amsgrad=False

#algo="LC_EGD"
algo="Adam"

Nb_tirages = 10

WEIGHTS = []
WEIGHTS_INIT=[]
for it in range(Nb_tirages):
  # Nouvelle initialisation
  w = 6 * np.random.random() - 3
  b = 6 * np.random.random() - 3
  WEIGHTS_INIT.append([w,b])
  params_init = [w, w, b, b]
  # Début du modèle
  model = build_poly(activation,loss,name_init,params_init,seed); 
  model_copy = build_poly(activation,loss,name_init,params_init,seed)
  # entrainement
  model, epoch, norme_grad, cost, temps = train(algo,model,model_copy,loss,x_train,y_train,eps,max_epochs,lr,seuil,f1,f2,rho,eps_egd,lambd,beta_1,beta_2,epsilon,amsgrad)
  print(it, model.get_weights())
  WEIGHTS.append(model.get_weights())
  print("temps: ", temps)

# Pour polyThree
glob=[[0,-1]]
loca=[[-2,1],[2,-1],[0,1]]
sadd=[[-1,0],[1,0],[-1,1],[1,-1]]

# Pour classer les points
seuil_acceptation=1.e-4

# Construction du code couleur
res=[]
i=0
for w in WEIGHTS:
  #print(w, w[0][0][0],w[1][0])
  Wl = np.array([w[0][0][0],w[1][0]])
  print(Wl)
  for g in glob:
    print(Wl, g, np.linalg.norm(Wl-g))
    if np.linalg.norm(Wl-g) < seuil_acceptation:
      res.append(0)
      print("GLOBAL")
      break
  for l in loca:
    #print(Wl, l,np.linalg.norm(Wl-l)[0])
    if np.linalg.norm(Wl-l) < seuil_acceptation:
      res.append(1)
      print("LOCAL")
      break
  for s in sadd:
    #print(Wl, s,np.linalg.norm(Wl-s)[0])
    if np.linalg.norm(Wl-s) < seuil_acceptation:
      res.append(2)
      print("SADDLE")
      break
  if len(res) < i:
    res.append(4)
  i+=1

code=["blue", "green", "orange", "yellow", "red"]
sign=["Global", "Local", "Point Selle", "à définir", "Non convergence"]
for i in range(len(sign)):
  print(WEIGHTS_INIT[i])
  plt.scatter(WEIGHTS_INIT[i][0], WEIGHTS_INIT[i][1],color=code[i], label=sign[i])
for i in range(Nb_tirages-1):
  plt.scatter(WEIGHTS_INIT[i][0], WEIGHTS_INIT[i][1],color=code[res[i]])
plt.legend()
plt.xlabel("b")
plt.ylabel("w")
plt.axis('equal')
plt.show()

