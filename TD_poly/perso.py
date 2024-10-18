from cmath import isnan
from tensorflow.keras import optimizers
import tensorflow as tf
import time
import numpy as np

from classic import Adam
from essai import LC_EM

def LC_EGD(model,model_copy,loss_fn,
x,y, eps, max_epochs, lr=0.1, f1=2, f2=10000, lambd=0.5, sample_weight=1):

    optimizer = optimizers.SGD(lr)
    norme_grad=1000; epoch=0; active_Adam=False; epoch_Adam=0; time_Adam=0;
    start_time = time.time()
    while(norme_grad>eps and epoch<max_epochs):

        if(epoch==0):
            with tf.GradientTape() as tape:

                prediction = model(x, training=True)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                cost = loss_fn(y, prediction,sample_weight=sample_weight)
                print("cost_init: ", cost)

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(cost, model.trainable_weights)
            norme_grad = tf.linalg.global_norm(grads); V_dot = norme_grad**2;
            if(norme_grad<eps):
                break
            print("grad_init: ", norme_grad)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.

        cost_prec = cost
        model_copy.set_weights(model.get_weights())
        condition=True
        iterLoop=0
        while(condition):
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            with tf.GradientTape() as tape:
                prediction = model(x, training=True)
                cost = loss_fn(y, prediction, sample_weight=sample_weight)
            condition = cost-cost_prec>-lambd*lr*V_dot
            if(condition):
                lr/=f1; optimizer.learning_rate=lr
                model.set_weights(model_copy.get_weights())
            iterLoop+=1
        #print("iterLoop: ", iterLoop)

        if(lr<10**(-10)):
            active_Adam=True
            break

        lr*=f2; optimizer.learning_rate=lr

        grads = tape.gradient(cost, model.trainable_weights)
        norme_grad= tf.linalg.global_norm(grads); V_dot=norme_grad**2

        epoch+=1    

        if epoch % 100 == 0:
            print("\nStart of epoch %d" % (epoch,))
            print(
                "Training loss (for one batch) at epoch %d: %.8f"
                % (epoch, float(cost))
            )
            print("grad: ", norme_grad)
            print("lr: ", lr)
            print("dim: ", cost-cost_prec)
            print('top: ', -lambd*lr*V_dot)

    """ print("epochs: ", epoch)
    print("grad_norm: ", norme_grad)
    print("cost_final: ", cost) """

    end_time = time.time()

    if(active_Adam==True):
        print("Pb conditionnement")
        #model,epoch_Adam, norme_grad,cost,time_Adam = Adam(model,loss_fn,x,y,eps,max_epochs-epoch)

    return model, epoch+epoch_Adam, norme_grad, cost, end_time-start_time+time_Adam

def LC_EGD2(model,model_copy,loss_fn,
x,y, eps, max_epochs, lr=0.1, f1=30, f2=10000, lambd=0.5,sample_weight=1):

    optimizer = optimizers.SGD(lr)
    limit_lr=10**(-4)
    norme_grad=1000; epoch=0; active_EGD3=False; epoch_EGD3=0; time_EGD3=0
    start_time = time.time()
    while(norme_grad>eps and epoch<max_epochs):

        if(epoch==0):
            with tf.GradientTape() as tape:

                prediction = model(x, training=True)

                # Compute the loss value for this minibatch.
                cost = loss_fn(y, prediction,sample_weight=sample_weight)
                print("cost_init: ", cost)

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(cost, model.trainable_weights)
            norme_grad = tf.linalg.global_norm(grads); V_dot = norme_grad**2;
            if(norme_grad<eps):
                break
            #print("grad_init: ", norme_grad)

        cost_prec = cost
        model_copy.set_weights(model.get_weights())
        condition=True
        iterLoop=0

        nbLoops=3
        while(condition):
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            with tf.GradientTape() as tape:
                prediction = model(x, training=True)
                cost = loss_fn(y, prediction, sample_weight=sample_weight)
            condition = cost-cost_prec>-lambd*lr*V_dot
            if(condition):
                lr/=f1; optimizer.learning_rate=lr
                model.set_weights(model_copy.get_weights())
            iterLoop+=1

        #dichotomie en échelle log
        if(iterLoop>1):
            droite = np.log10(lr*f1); gauche = np.log10(lr)
            for k in range(nbLoops):
                m = (gauche+droite)/2; lr=10**m; optimizer.learning_rate = lr
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                with tf.GradientTape() as tape:
                    prediction = model(x, training=True)
                    cost = loss_fn(y, prediction, sample_weight=sample_weight)
                if(cost-cost_prec>-lambd*lr*V_dot):
                    m_best = gauche
                    droite = m
                    last_pass=False
                else:
                    gauche = m
                    last_pass=True
                if(k<nbLoops-1):
                    model.set_weights(model_copy.get_weights())
                else:
                    if(last_pass==False):
                        lr=10**m_best; optimizer.learning_rate = lr
                        model.set_weights(model_copy.get_weights())
                        optimizer.apply_gradients(zip(grads, model.trainable_weights))
                        with tf.GradientTape() as tape:
                            prediction = model(x, training=True)
                            cost = loss_fn(y, prediction, sample_weight=sample_weight)
                iterLoop+=1
        #print("iterLoop: ", iterLoop)

        """ if(lr<limit_lr):
            active_EGD3=True; break """

        lr*=f2; optimizer.learning_rate=lr

        grads = tape.gradient(cost, model.trainable_weights)
        norme_grad= tf.linalg.global_norm(grads); V_dot=norme_grad**2

        epoch+=1    

        if epoch % 2 == 0:
            print("\nStart of epoch %d" % (epoch,))
            print(
                "Training loss (for one batch) at epoch %d: %.8f"
                % (epoch, float(cost))
            )
            print("grad: ", norme_grad)
            print("lr: ", lr)
            print("dim: ", cost-cost_prec)
            print('top: ', -lambd*lr*V_dot)
        
    if(active_EGD3):
        print("LC_EGD débute")
        model,epoch_EGD3, norme_grad,cost,time_EGD3= LC_EGD(model,model_copy,loss_fn,x,y,eps,max_epochs-epoch,2,100)

    """ print("epochs: ", epoch)
    print("grad_norm: ", norme_grad)
    print("cost_final: ", cost) """

    end_time = time.time()

    return model, epoch+epoch_EGD3, norme_grad, cost, end_time-start_time+time_EGD3

def LC_EGD3(model,model_copy,loss_fn,
x,y, eps, max_epochs, lr=0.1, rho=0.9, epsilon=0.01, lambd=0.5,sample_weight=1):

    optimizer = optimizers.SGD(lr)
    norme_grad=1000; epoch=0; active_Adam=False; epoch_Adam=0; time_Adam=0
    start_time = time.time()
    while(norme_grad>eps and epoch<max_epochs):

        if(epoch==0):
            with tf.GradientTape() as tape:
                prediction = model(x, training=True)  # Logits for this minibatch

                cost = loss_fn(y, prediction,sample_weight=sample_weight)
                #print("cost_init: ", cost)

            grads = tape.gradient(cost, model.trainable_weights)
            norme_grad = tf.linalg.global_norm(grads); V_dot = norme_grad**2;
            if(norme_grad<eps):
                break
            #print("grad_init: ", norme_grad)

        cost_prec = cost
        model_copy.set_weights(model.get_weights())
        condition=True
        iterLoop=0
        while(condition and iterLoop<500):
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            with tf.GradientTape() as tape:
                prediction = model(x, training=True)
                cost = loss_fn(y, prediction, sample_weight=sample_weight)
            condition = cost-cost_prec>-lambd*lr*V_dot
            if(condition):
                f1 = (rho*(1-lambd)*V_dot)/max((cost-cost_prec)/lr+V_dot,epsilon*(1-lambd)*V_dot)
                lr*=f1; optimizer.learning_rate=lr
                model.set_weights(model_copy.get_weights())
            iterLoop+=1
        #print("iterLoop: ", iterLoop)
        f1 = (rho*(1-lambd)*V_dot)/max((cost-cost_prec)/lr+V_dot,epsilon*(1-lambd)*V_dot)
        lr*=f1; optimizer.learning_rate=lr

        grads = tape.gradient(cost, model.trainable_weights)
        norme_grad= tf.linalg.global_norm(grads); V_dot=norme_grad**2

        if(isnan(norme_grad) or lr<10**(-10)):
            active_Adam=True; break

        epoch+=1    

        if epoch % 2 == 0:
            print("\nStart of epoch %d" % (epoch,))
            print(
                "Training loss (for one batch) at epoch %d: %.8f"
                % (epoch, float(cost))
            )
            print("grad: ", norme_grad)
            print("lr: ", lr)
            print("dim: ", cost-cost_prec)
            print('top: ', -lambd*lr*V_dot)

    """ print("epochs: ", epoch)
    print("grad_norm: ", norme_grad)
    print("cost_final: ", cost) """

    end_time = time.time()

    if(active_Adam==True):
        print("Active Adam")
        model,epoch_Adam, norme_grad,cost,time_Adam = Adam(model,loss_fn,x,y,eps,max_epochs-epoch)

    return model, epoch+epoch_Adam, norme_grad, cost, end_time-start_time+time_Adam

def ER(model,model_inter,loss_fn,
x,y, eps, max_epochs, lr=0.1, seuil=0.01, sample_weight=1):

    optimizer = optimizers.SGD(lr)
    norme_grad=1000; epoch=0
    model_inter.set_weights(model.get_weights())
    start_time = time.time()
    while(norme_grad>eps and epoch<max_epochs):

        if(epoch==0):
            with tf.GradientTape() as tape:
                prediction = model(x, training=True)  # Logits for this minibatch
                cost = loss_fn(y, prediction,sample_weight=sample_weight)
                print("cost_init: ", cost)

            grads = tape.gradient(cost, model.trainable_weights)
            norme_grad = tf.linalg.global_norm(grads)
            if(norme_grad<eps):
                return model,0
            print("grad_init: ", norme_grad)

            optimizer.apply_gradients(zip(grads, model_inter.trainable_weights))
            with tf.GradientTape() as tape:
                prediction = model_inter(x, training=True)  # Logits for this minibatch
                cost = loss_fn(y, prediction,sample_weight=sample_weight)
            grads_inter = tape.gradient(cost, model_inter.trainable_weights)
        
        e = (lr*tf.linalg.global_norm([a-b for a,b in zip(grads,grads_inter)]))/(2*seuil)

        if(e>1):
            lr*=0.9/np.sqrt(e); optimizer.learning_rate=lr
        else:
            optimizer.apply_gradients(zip(grads_inter, model.trainable_weights))
            with tf.GradientTape() as tape:
                prediction = model(x, training=True)  # Logits for this minibatch
                cost = loss_fn(y, prediction,sample_weight=sample_weight)
            grads = tape.gradient(cost, model.trainable_weights)
            lr*=0.9/np.sqrt(e); optimizer.learning_rate=lr
        
        model_inter.set_weights(model.get_weights())
        optimizer.apply_gradients(zip(grads, model_inter.trainable_weights))
        with tf.GradientTape() as tape:
                prediction = model_inter(x, training=True)  # Logits for this minibatch
                cost = loss_fn(y, prediction,sample_weight=sample_weight)
        grads_inter = tape.gradient(cost, model_inter.trainable_weights)
        norme_grad = tf.linalg.global_norm(grads_inter)

        epoch+=1

        if epoch % 2 == 0:
            print("\nStart of epoch %d" % (epoch,))
            print(
                "Training loss (for one batch) at epoch %d: %.8f"
                % (epoch, float(cost))
            )
            print("grad: ", norme_grad)
            print("lr: ", lr)
            #print("Seen so far: %s samples" % ((step + 1) * batch_size))

    print("epochs: ", epoch)
    print("grad_norm: ", norme_grad)
    print("cost_final: ", cost)

    end_time = time.time()

    return model_inter, epoch, norme_grad, cost, end_time-start_time



def LC_EGD_Adam(model,model_copy,loss_fn,
x,y, eps, max_epochs, lr=0.1, f1=30, f2=10000, lambd=0.5,sample_weight=1):

    optimizer = optimizers.SGD(lr)
    norme_grad=1000; epoch=0; active_Adam=False; epoch_Adam=0; time_Adam=0
    start_time = time.time()
    while(norme_grad>eps and epoch<max_epochs):

        if(epoch==0):
            with tf.GradientTape() as tape:

                prediction = model(x, training=True)

                cost = loss_fn(y, prediction,sample_weight=sample_weight)
                print("cost_init: ", cost)

            grads = tape.gradient(cost, model.trainable_weights)
            norme_grad = tf.linalg.global_norm(grads); V_dot = norme_grad**2;
            if(norme_grad<eps):
                break
            #print("grad_init: ", norme_grad)

        cost_prec = cost
        model_copy.set_weights(model.get_weights())
        condition=True
        iterLoop=0

        nbLoops=3
        while(condition):
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            with tf.GradientTape() as tape:
                prediction = model(x, training=True)
                cost = loss_fn(y, prediction, sample_weight=sample_weight)
            condition = cost-cost_prec>-lambd*lr*V_dot
            if(condition):
                lr/=f1; optimizer.learning_rate=lr
                model.set_weights(model_copy.get_weights())
            iterLoop+=1

        #dichotomie en échelle log
        if(iterLoop>1):
            droite = np.log10(lr*f1); gauche = np.log10(lr)
            for k in range(nbLoops):
                m = (gauche+droite)/2; lr=10**m; optimizer.learning_rate = lr
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                with tf.GradientTape() as tape:
                    prediction = model(x, training=True)
                    cost = loss_fn(y, prediction, sample_weight=sample_weight)
                if(cost-cost_prec>-lambd*lr*V_dot):
                    m_best = gauche
                    droite = m
                    last_pass=False
                else:
                    gauche = m
                    last_pass=True
                if(k<nbLoops-1):
                    model.set_weights(model_copy.get_weights())
                else:
                    if(last_pass==False):
                        lr=10**m_best; optimizer.learning_rate = lr
                        model.set_weights(model_copy.get_weights())
                        optimizer.apply_gradients(zip(grads, model.trainable_weights))
                        with tf.GradientTape() as tape:
                            prediction = model(x, training=True)
                            cost = loss_fn(y, prediction, sample_weight=sample_weight)
                iterLoop+=1
        #print("iterLoop: ", iterLoop)

        if(lr<10**(-4) or (cost_prec-cost)/cost<0.001):
            active_Adam=True; break

        lr*=f2; optimizer.learning_rate=lr

        grads = tape.gradient(cost, model.trainable_weights)
        norme_grad= tf.linalg.global_norm(grads); V_dot=norme_grad**2

        epoch+=1    

        """ if epoch % 2 == 0:
            print("\nStart of epoch %d" % (epoch,))
            print(
                "Training loss (for one batch) at epoch %d: %.8f"
                % (epoch, float(cost))
            )
            print("grad: ", norme_grad)
            print("lr: ", lr)
            print("dim: ", cost-cost_prec)
            print('top: ', -lambd*lr*V_dot) """
        
    if(active_Adam):
        print("Adam débute")
        model,epoch_Adam, norme_grad,cost,time_Adam= Adam(model,loss_fn,x,y,eps,max_epochs-epoch)

    """ print("epochs: ", epoch)
    print("grad_norm: ", norme_grad)
    print("cost_final: ", cost) """

    end_time = time.time()

    return model, epoch+epoch_Adam, norme_grad, cost, end_time-start_time+time_Adam
