#Labo 6 - Funcion Sigmoidea

import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore


def sigmoid(z):
    """
    Compute the sigmoid of z
    Args:
        z (ndarray): A scalar, numpy array of any size.
    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
    """
    g = 1/(1+np.exp(-z))

    return g


X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  #(m,n)
y_train = np.array([0, 0, 0, 1, 1, 1])        


def plot_data(X,y):
  X_clasificado_0 =[]
  X_clasificado_1 =[]
  for i in range(len(y_train)):
    if y_train[i] == 0:
      X_nuevo_clas_0 = (X_train[i][0] , X_train[i][1])
      X_clasificado_0.append(X_nuevo_clas_0)
    if y_train[i] == 1:
      X_nuevo_clas_1 = (X_train[i][0] , X_train[i][1])
      X_clasificado_1.append(X_nuevo_clas_1)

  # Unpack the list of tuples into two lists for x and y coordinates
  x_coords_0 = [p[0] for p in X_clasificado_0]
  y_coords_0 = [p[1] for p in X_clasificado_0]

  x_coords_1 = [p[0] for p in X_clasificado_1]
  y_coords_1 = [p[1] for p in X_clasificado_1]
  return x_coords_0, y_coords_0, x_coords_1, y_coords_1

x_coords_0, y_coords_0, x_coords_1, y_coords_1 = plot_data(X_train,y_train)


def plot_data(X,y):
  X_clasificado_0 =[]
  X_clasificado_1 =[]
  for i in range(len(y_train)):
    if y_train[i] == 0:
      X_nuevo_clas_0 = (X_train[i][0] , X_train[i][1])
      X_clasificado_0.append(X_nuevo_clas_0)
    if y_train[i] == 1:
      X_nuevo_clas_1 = (X_train[i][0] , X_train[i][1])
      X_clasificado_1.append(X_nuevo_clas_1)

  # Unpack the list of tuples into two lists for x and y coordinates
  x_coords_0 = [p[0] for p in X_clasificado_0]
  y_coords_0 = [p[1] for p in X_clasificado_0]

  x_coords_1 = [p[0] for p in X_clasificado_1]
  y_coords_1 = [p[1] for p in X_clasificado_1]
  return x_coords_0, y_coords_0, x_coords_1, y_coords_1

x_coords_0, y_coords_0, x_coords_1, y_coords_1 = plot_data(X_train,y_train)

def compute_cost_logistic(X, y, w, b):
    """
    Computes cost

    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter

    Returns:
      cost (scalar): cost
    """

    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i],w) + b
        f_wb_i = sigmoid(z_i)
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i) #funcion de costos simplificada para problemas de tipo clasificacion

    cost = cost / m
    return cost

def compute_gradient_logistic(X, y, w, b):
    """
    Computes the gradient for logistic regression

    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters
      b (scalar)      : model parameter
    Returns
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b.
    """
    m,n = X.shape
    dj_dw = np.zeros((n,))                           #(n,)
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)          #(n,)(n,)=scalar
        err_i  = f_wb_i  - y[i]                       #scalar
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]      #scalar
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m                                   #(n,)
    dj_db = dj_db/m                                   #scalar

    return dj_db, dj_dw


def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    """
    Performs batch gradient descent

    Args:
      X (ndarray (m,n)   : Data, m examples with n features
      y (ndarray (m,))   : target values
      w_in (ndarray (n,)): Initial values of model parameters
      b_in (scalar)      : Initial values of model parameter
      alpha (float)      : Learning rate
      num_iters (scalar) : number of iterations to run gradient descent

    Returns:
      w (ndarray (n,))   : Updated values of parameters
      b (scalar)         : Updated value of parameter
    """
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in

    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion
            J_history.append( compute_cost_logistic(X, y, w, b) )

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")

    return w, b, J_history         #return final w,b and J history for graphing


#Pruebas:


X_tmp = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_tmp = np.array([0, 0, 0, 1, 1, 1])
dj_db_tmp, dj_dw_tmp = compute_gradient_logistic(X_tmp, y_tmp, w_tmp, b_tmp)

w_tmp  = np.zeros_like(X_train[0])
b_tmp  = 0.
alph = 0.1
iters = 10000

w_out, b_out, _ = gradient_descent(X_train, y_train, w_tmp, b_tmp, alph, iters)
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")

fig,ax = plt.subplots(1,1,figsize=(5,4))
# plot the probability

ax.axis([0, 4, 0, 4])

x_coords_0, y_coords_0, x_coords_1, y_coords_1 =plot_data(X_train,y_train)

ax.scatter(x_coords_0, y_coords_0, marker = "x", c="r", label="y = 0")
ax.scatter(x_coords_1, y_coords_1, marker = "o", c= "b", label="y =  1")

ax.set_title("Clasificación de Datos")
ax.set_xlabel("x0")
ax.set_ylabel("x1")
ax.legend()
ax.grid(True)


# Plot the original data
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')


# Plot the decision boundary
x0 = -b_out/w_out[0]
x1 = -b_out/w_out[1]
ax.plot([0,x0],[x1,0], c='blue', lw=1)
plt.show()