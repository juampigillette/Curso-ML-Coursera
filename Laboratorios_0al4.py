
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

x_train = np.array([1.0, 2.0])   #features
y_train = np.array([300.0, 500.0])   #target value

def funcion_lineal(x_modelo,pendiente_m,odo_b):
  return pendiente_m *x_modelo+ odo_b

def compute_cost(x, y, w, b):
   
    m = x.shape[0] 
    cost = 0
    
    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2
    total_cost = 1 / (2 * m) * cost

    return total_cost

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
    dj_db = 0               #inicializo b=0

    for i in range(m): #iteracion sobre los ejemplos
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n): #iteracion sobre los features dentro de un mismo ejemplo
            dj_dw[j] = dj_dw[j] + err * X[i, j] #actualizo valores con error
        dj_db = dj_db + err
    dj_dw = dj_dw / m #finalizo update
    dj_db = dj_db / m #finalizo update

    return dj_db, dj_dw

def gradient_descent(x, y, w_inicial, b_inicial, alpha, num_iters, cost_function, gradient_function): 
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x (ndarray (m,))  : Data, m examples 
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): valores iniciales w y b arbitrarios
      alpha (float):     Learning rate
      num_iters (int):   numero de iteraciones para actualizar los parametros
      cost_function:     funcion de costos
      gradient_function: funcion a llamar que produce el los gradientes
      
    Returns:
      w (scalar): valor w actualizado, post algoritmo de gradiente descendiente
      b (scalar): valor b actualizado, post algoritmo de gradiente descendiente
      J_history (List): Historia de los valores de costo J(w,b)
      p_history (list): Historia de los parametros w y b
      """
    
    J_history = [] 
    p_history = []
    b = b_inicial
    w = w_inicial
  
    for i in range(num_iters):
        # Calculamos el gradiente para poder utilizarlo en la actualizacion de los parametros
        dj_dw, dj_db = gradient_function(x, y, w , b)     

        # Actualizo parametros
        b = b - alpha * dj_db                            
        w = w - alpha * dj_dw                            

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(x, y, w , b))
            p_history.append([w,b])

    return w, b, J_history, p_history #return w and J,w history for graphing

#Pruebas: 

x_train = np.array([1.0, 2.0, 3.0 , 4.0, 5.0, 6.0, 7.0 , 8.0, 9.0])   #features
y_train = np.array([300.0, 500.0, 700.0 , 850.0 , 1000.0, 1250.0 , 1350.0 , 1500.0 , 1700.0])   #target value


w_inicial = 0
b_inicial = 0
# configuracion de gradiente descendente dadas por el curso
iterations = 10000
alpha = 1.0e-2

# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_inicial, b_inicial, alpha, 
                                                    iterations, compute_cost, compute_gradient)

for i in range (0,10000,1000): #para ver la progresion de J
  print(f"valor J en iteracion {i} : {J_hist[i]:0.2e}")

print(f"valor J final {J_hist[iterations-1]:0.2e}") #para ver la progresion de J
print ("*"*100)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")

plt.scatter(x_train, y_train, marker = "x", c="b")
plt.plot(x_train, funcion_lineal(x_train,w_final,b_final), c = "r", label = "regresion lineal")
plt.title("Ejemplo de juampi")
plt.ylabel("titulo de las y")
plt.xlabel("titulo de las x")
plt.show()
