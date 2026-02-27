
import numpy as np
import matplotlib.pyplot as plt


def predict_single_loop(x, w, b): #con w y b definidos.
    """
    single predict using linear regression

    Args:
      x (ndarray): Shape (n,) ejemplo con multiples caracteristicas (features)
      w (ndarray): Shape (n,) parametros (n) del modelo
      b (scalar):  parametro bias

    Returns:
      p (scalar):  prediccion
    """
    n = x.shape[0]
    p = 0
    for i in range(n):
        p_i = x[i] * w[i]
        p = p + p_i
    p = p + b
    return p

#Forma vectorizada:

def predict(x, w, b):
    """
    single predict using linear regression
    Args:
      x (ndarray): Shape (n,) ejemplo con multiples caracteristicas (features)
      w (ndarray): Shape (n,) parametros (n) del modelo
      b (scalar):  parametro bias

    Returns:
      p (scalar):  prediccion
    """
    p = np.dot(x, w) + b     #utilizamos producto escalar de NumPy
    return p

  #Calculo del costo:
def compute_cost(X, y, w, b):
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m ejemplos con n features
      y (ndarray (m,)) : valores objetivos (targets)
      w (ndarray (n,)) :  parametros (n) del modelo
      b (scalar)       : parametro bias

    Returns:
      cost (scalar): costo
    """
    m = X.shape[0]
    cost = 0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b           #(n,) . (n,) + b = escalar
        cost = cost + (f_wb_i - y[i])**2
    cost = cost / (2 * m)
    return cost



def compute_gradient(X, y, w, b):
    """
    Computes the gradient for linear regression
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter

    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b.
    """
    m,n = X.shape           #(m ejemplos, n features)
    dj_dw = np.zeros((n,))   #inicializo w =[0,0,0..0]
    dj_db = 0.               #inicializo b=0

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j] #actualizo valores con error
        dj_db = dj_db + err
    dj_dw = dj_dw / m #finalizo update
    dj_db = dj_db / m #finalizo update

    return dj_db, dj_dw


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Hace un batch gradient descent para encontrar w y b. Actualiza w y b tomando num_iters gradientes con paso alpha.


    Argumentos:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w_in (ndarray (n,)) : parametros w modelo iniciales
      b_in (scalar)       : paramtro b modelo inicial
      cost_function       : funcion de costo
      gradient_function   : funcion de calculo de gradiente
      alpha (float)       : Learning rate
      num_iters (int)     : numero de iteraciones

    Returns:
      w (ndarray (n,)) : valores w actualizados
      b (scalar)       : valor b actualizado
      """

    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = w_in #hacemos una copia de w para no modificar el w_in
    b = b_in

    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)   

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               

    return w, b, J_history #return final w,b and J history for graphing


#Prueba:

X_train = np.array([[2104, 5, 1, 45], #Ejemplo 1
                    [1416, 3, 2, 40], #Ejemplo 2
                     [852, 2, 1, 35]]) #xEjemplo 3
#X es una matriz de 3x4, con 3 ejemplos, cada uno con 4 features correspondientes

y_train = np.array([460, 232, 178]) #Valores de output para cada ejemplo

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

# initialize parameters
initial_w = np.zeros_like(w_init)
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 5.0e-7
# run gradient descent 
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient, 
                                                    alpha, iterations)

print(f"b,w hallado con gradiente descendiente: {b_final:0.2f},{w_final} ")
m,_ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")