import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

def solverFVM( x,y, rho, gamma, n, L, a,discret):
    
    h = L/n     
    if discret == 'uds':

        A = np.zeros((n*n, n*n), dtype=np.float32)
        b = np.zeros(n*n, dtype=np.float32)
        lenghr = len(A)
        print(f"{lenghr}")
        
        for i in range(1, n*n-1):
            for j in range(1, n*n-1):
                x = (j%n)*h
                y = (i%n)*h
                
                c_P = (rho*x + 2*gamma/h +rho*h)*h + (rho*y + 2*gamma/h)*h
                c_W = (-rho*x - gamma/h)*h
                c_E = (-gamma/h)*h
                c_S = (-gamma/h)*h
                c_N = (-rho*y -gamma/h -rho*y)*h
                RH = ((2*x*h+h**2)*(2*y*h+h**2))/4 
                
                if n*(n-1)>i and i>n-1 and n*(n-1)>j and j>n-1 and i!=n*i and j!=n*j and i==j:
                    A[i,j] = c_P
                    A[i,j+1] = c_E
                    A[i,j-1] = c_W
                    A[n*((i+1)%n),j%n] = c_N
                    A[n*((i-1)%n),j%n] = c_S
                    b[(n*(i%n))+(j%n)] = RH        

        for i in range(n*n-n+1, n*n-1):
            x = (j%n)*h
            y = n-1
                    
            c_P=(rho*x + 2*gamma/h +rho*h)*h + (rho*y + 3*gamma/h)*h
            c_W = (-rho*x-gamma/h)*h
            c_S= (-gamma/h)*h
            c_E= (-gamma/h)*h
            RH_T = ((2*x*h+h**2)*(2*y*h+h**2))/4
            
            A[i, i] = c_P
            A[i, i+1] = c_E
            A[i, i-1] = c_W
            A[i-n,i] = c_S
            b[i] = RH_T
    

        j = n-1
        for i in range(1,n*n-1):
            x = n-1
            y = (i%n)*h
            
            if i%n==n-1:
                    
                c_P = (rho*x + gamma/h +rho*h)*h + (rho*y + 2*gamma/h)*h
                c_W= (-rho*x - gamma/h)*h
                c_N= (-rho*y -gamma/h - rho*h)*h
                c_S= (-gamma/h)*h
                RH_R= ((2*x*h+h**2)*(2*y*h+y**2))/4
                
                A[int((i+1)/n)*n-1, int((i+1)/n)*n-1] = c_P
                A[int((i+1)/n)*n-1, int((i+1)/n)*n-1-1] = c_W
                A[int((i+1)/n)*n-1-n,int((i+1)/n)*n-1] = c_S
                A[int((i+1)/n)*n-1+n,int((i+1)/n)*n-1] = c_N
                b[int((i+1)/n)*n-1] = RH_R


        j = 0
        for i in range(1,n*n-1):
            x = 0
            y = (i%n)*h
            
            if i%n==0:
                    
                c_P = (rho*x + 3*gamma/h +rho*h)*h + (rho*y + 2*gamma/h)*h
                c_N= (-rho*y -rho*h -gamma/h)*h
                c_S= (-gamma/h)*h
                c_E= (-gamma/h)*h
                RH_L = ((2*x*h+h**2)*(2*y*h+h**2))/4 + (1-y)*rho*x*h + (gamma*(2-2*y)*h)/h
                
                A[(n*int(i/n)), (n*int(i/n))] = c_P
                A[(n*int(i/n)), (n*int(i/n))+1] = c_E
                A[n*int((i-1)/n),(n*int(i/n))] = c_S
                A[n*int((i+1)/n),(n*int(i/n))] = c_N
                b[(n*int(i/n))] = RH_L
            

        i = 0
        for j in range(1,n-1):
            x = (j)*h
            y = 0
            
            c_W = (-rho*x - gamma/h)*h
            c_P= (rho*x + 2*gamma/h +rho*h)*h + (rho*y + gamma/h)*h
            c_E= (-gamma/h)*h
            c_N= (-rho*y -rho*h -gamma/h)*h
            
            if (x<0.2) or (x>=0.8):
                RH_B1= ((2*x*h+h**2)*(2*y*h+y**2))/4
                A[(n*i+j),(n*i+j)] = c_P
                A[(n*i+j),(n*i+j)+1] = c_E
                A[(n*i+j),(n*i+j)-1] = c_W
                A[n*(i+1),(n*i+j)] = c_N
                b[(n*i)+j] = RH_B1
            elif (0.2<=x<0.8):
                RH_B2= ((2*x*h+h**2)*(2*y*h+y**2))/4 - a*h + (y*a*h*h)/(2*gamma)
                A[(n*i+j),(n*i+j)] = (rho*x + 2*gamma/h +rho*h)*h + (rho*y + gamma/h)*h
                A[(n*i+j),(n*i+j)+1] = (-gamma/h)*h
                A[(n*i+j),(n*i+j)-1] = (-rho*x - gamma/h)*h
                A[n*(i+1),j] = (-rho*y -rho*h -gamma/h)*h
                b[(n*i)+j] = RH_B2    


        x = 0
        y = 0
        c_P_LL = (rho*x + 3*gamma/h +rho*h)*h +( rho*y + gamma/h)*h
        c_E_LL = (-gamma/h)*h
        c_N_LL = (-rho*y -gamma/h-rho*h)*h
        RH_LL = ((2*x*h+h**2)*(2*y*h+y**2))/4 + rho*x*(1-y)*h + gamma*(2-2*y)
        
        A[0,0] = c_P_LL
        A[0,1] = c_E_LL
        A[n,0] = c_N_LL
        b[0] = RH_LL
   
        x = 0
        y = (n-1)*h
        c_P_LT= (rho*x + 3*gamma/h +rho*h)*h +( rho*y + 3*gamma/h)*h
        c_E_LT = (-gamma/h)*h
        c_S_LT= -gamma
        RH_LT = ((2*x*h+h**2)*(2*y*h+y**2))/4 
        
        A[n*(n-1),n*(n-1)] = c_P_LT
        A[n*(n-1),n*(n-1)+1] = c_E_LT
        A[n*(n-2),n*(n-1)] = c_S_LT
        b[n*(n-1)] = RH_LT
   

        x = (n-1)*h
        y = (n-1)*h
        c_P_RT= (rho*x + gamma/h +rho*h)*h +( rho*y + 3*gamma/h)*h
        c_W_RT = (-rho*x - gamma/h)*h
        c_S_RT= -gamma
        RH_LT = ((2*x*h+h**2)*(2*y*h+y**2))/4 
        
        A[n*n-1, n*n-1] = c_P_RT
        A[n*n-1, n*n-2] = c_W_RT
        A[n*n-1-n, n*n-1] = c_S_RT
        b[n*n-1] = RH_LT
   
        x = (n-1)*h
        y = 0
        c_P_RL= (rho*x + gamma/h +rho*h)*h +( rho*y + gamma/h)*h
        c_W_RL = (-rho*x - gamma/h)*h
        c_N_RL = (-rho*y -(gamma/h)-rho*h)*h
        RH_RL = ((2*x*h+h**2)*(2*y*h+y**2))/4 
        
        A[n-1,n-1] = c_P_RL
        A[n-1,n-2] = c_W_RL
        A[2*n-1,n-1] = c_N_RL
        b[n-1] = RH_RL


    elif discret == 'cds':
        A = np.zeros((n*n, n*n), dtype=np.float32)
        b = np.zeros(n*n, dtype=np.float32)
        lenghr = len(A)
        print(f"{lenghr}")
        for i in range(1, n*n-1):
            for j in range(1, n*n-1):
                x = (j%n)*h
                y = (i%n)*h


                c_P = ( 2*gamma/h)*h + (2*gamma/h)*h
                c_W = (-rho*x - gamma/h)*h
                c_E = (rho*x*0.5 +rho*h*0.5-gamma/h)*h
                c_S = (rho*y*h*0.5 - gamma/h)*h
                c_N = (-rho*y*0.5 -gamma/h -rho*h*0.5 )*h
                RH = ((2*x*h+h**2)*(2*y*h+y**2))/4

                if n*(n-1)>i and i>n-1 and n*(n-1)>j and j>n-1 and i!=n*i and j!=n*j and i==j:
                    A[i,j] = c_P
                    A[i,j+1] = c_E
                    A[i,j-1] = c_W
                    A[n*((i+1)%n),j%n] = c_N
                    A[n*((i-1)%n),j%n] = c_S
                    b[(n*(i%n))+(j%n)] = RH        
        
        for i in range(n*n-n+1, n*n-1):
            x = (j%n)*h
            y = n-1
                    
            
            c_P=( 2*gamma/h +rho*h*0.5)*h + (rho*y*0.5 + 3*gamma/h)*h
            c_W = (-rho*x*0.5 -gamma/h)*h
            c_S= (rho*y*0.5-gamma/h)*h
            c_E= (rho*x*0.5 -rho*h*0.5 -gamma/h)*h
            RH_T = ((2*x*h+h**2)*(2*y*h+h**2))/4
            
            A[i, i] = c_P
            A[i, i+1] = c_E
            A[i, i-1] = c_W
            A[i-n,i] = c_S
            b[i] = RH_T
        
        j = n-1
        for i in range(1,n*n-1):
            x = n-1
            y = (i%n)*h
            if i%n==n-1:
                c_P = (rho*x*0.5 + gamma/h +rho*h*0.5)*h + (2*gamma/h)*h
                c_W = (-rho*x*0.5 - gamma/h)*h
                c_N= (-rho*y*0.5 -gamma/h - rho*h*0.5)*h
                c_S= (rho*y*0.5 - gamma/h)*h
                RH_R= ((2*x*h+h**2)*(2*y*h+y**2))/4

                A[int((i+1)/n)*n-1, int((i+1)/n)*n-1] = c_P
                A[int((i+1)/n)*n-1, int((i+1)/n)*n-1-1] = c_W
                A[int((i+1)/n)*n-1-n,int((i+1)/n)*n-1] = c_S
                A[int((i+1)/n)*n-1+n,int((i+1)/n)*n-1] = c_N
                b[int((i+1)/n)*n-1] = RH_R


        j = 0
        for i in range(1,n*n-1):
            x = 0
            y = (i%n)*h
            
            if i%n==0:
                c_P = (rho*x*0.5 + 3*gamma/h )*h + (2*gamma/h)*h
                c_N= (-rho*y*0.5 -rho*h*0.5 +gamma/h)*h
                c_S= (rho*y*0.5 + gamma/h)*h
                c_E= (rho*x*0.5+rho*h*0.5-gamma/h)*h
                RH_L = ((2*x*h+h**2)*(2*y*h+h**2))/4 + (1-y)*rho*x*h + (gamma*(2-2*y)*h)/h
                
                A[(n*int(i/n)), (n*int(i/n))] = c_P
                A[(n*int(i/n)), (n*int(i/n))+1] = c_E
                A[n*int((i-1)/n),(n*int(i/n))] = c_S
                A[n*int((i+1)/n),(n*int(i/n))] = c_N
                b[(n*int(i/n))] = RH_L

        i = 0
        for j in range(1,n-1):
            x = (j)*h
            y = 0
            c_W = (-rho*x*0.5 - gamma/h)*h
            c_P = (rho*x*0.5 + 2*gamma/h )*h + ( gamma/h)*h
            c_E= (rho*x*0.5+rho*h*0.5-gamma/h)*h
            c_N= (-rho*y*0.5 -rho*h*0.5 -gamma/h)*h
            


            if (x<0.2) or (x>=0.8):
                RH_B1= ((2*x*h+h**2)*(2*y*h+y**2))/4
                A[(n*i+j),(n*i+j)] = c_P
                A[(n*i+j),(n*i+j)+1] = c_E
                A[(n*i+j),(n*i+j)-1] = c_W
                A[n*(i+1),(n*i+j)] = c_N
                b[(n*i)+j] = RH_B1

            elif (0.2<=x<0.8):
                RH_B2= ((2*x*h+h**2)*(2*y*h+y**2))/4 - a*h +(rho*h*h*y*a)/2*gamma
                A[(n*i+j),(n*i+j)] = ( 2*gamma/h )*h + (rho*y*0.5 + gamma/h)*h
                A[(n*i+j),(n*i+j)+1] = (rho*x*0.5+rho*h*0.5 -gamma/h)*h
                A[(n*i+j),(n*i+j)-1] = (-rho*x*0.5 - gamma/h)*h
                A[n*(i+1),j] = (-rho*y*0.5 -rho*h*0.5 -gamma/h)*h
                b[(n*i)+j] = RH_B2 

        x = 0
        y = 0
        c_P_LL = (rho*x*0.5 + 3*gamma/h )*h +( rho*y*0.5 + gamma/h)*h
        c_E_LL = (rho*x*0.5 + rho*h*0.5 - gamma/h)*h
        c_N_LL = (-rho*y*0.5 -gamma/h -rho*h*0.5)*h
        RH_LL = ((2*x*h+h**2)*(2*y*h+y**2))/4 + rho*x*(1-y)*h + gamma*(1-1*y)

        
        A[0,0] = c_P_LL
        A[0,1] = c_E_LL
        A[n,0] = c_N_LL
        b[0] = RH_LL

        x = 0
        y = (n-1)*h
        c_P_LT= (rho*x*0.5 + 3*gamma/h +rho*h*0.5)*h +( rho*y + 3*gamma/h)*h
        c_E_LT = (rho*x*0.5 + rho*h*0.5 -gamma/h)*h
        c_S_LT= rho*y*h*0.5 -gamma
        RH_LT = ((2*x*h+h**2)*(2*y*h+y**2))/4 

        
        A[n*(n-1),n*(n-1)] = c_P_LT
        A[n*(n-1),n*(n-1)+1] = c_E_LT
        A[n*(n-2),n*(n-1)] = c_S_LT
        b[n*(n-1)] = RH_LT

        
        x = (n-1)*h
        y = (n-1)*h
        c_P_RT= (rho*x*0.5 + gamma/h +rho*h*0.5)*h +( rho*y*0.5 + 3*gamma/h)*h
        c_W_RT = (-rho*x*0.5 - gamma/h)*h
        c_S_RT= rho*y*h*0.5 -gamma
        RH_LT = ((2*x*h+h**2)*(2*y*h+y**2))/4  
        
        A[n*n-1, n*n-1] = c_P_RT
        A[n*n-1, n*n-2] = c_W_RT
        A[n*n-1-n, n*n-1] = c_S_RT
        b[n*n-1] = RH_LT


        
        x = (n-1)*h
        y = 0
        c_P_RL= (rho*x*0.5 + gamma/h +rho*h*0.5)*h +( rho*y*0.5 + gamma/h)*h
        c_W_RL = (-rho*x*0.5 - gamma/h)*h
        c_N_RL = (-rho*y*0.5 -(gamma/h)-rho*h*0.5)*h
        RH_RL = ((2*x*h+h**2)*(2*y*h+y**2))/4 

        
        A[n-1,n-1] = c_P_RL
        A[n-1,n-2] = c_W_RL
        A[2*n-1,n-1] = c_N_RL
        b[n-1] = RH_RL

            
    print(f"{A}")
    
    sol = np.matmul(inv(A), b)
    return sol

L = 1
x = 0
y = 0
phi_u = 0
phi_l = 1-y
gamma = 0.01
rho = 1
n = 40
a = 0.1
discret = 'uds' 

phi_uds = solverFVM(x, y, rho, gamma, n, L, a,discret)
print(phi_uds)

phi_2d = phi_uds.reshape((n, n))
plt.figure(figsize=(8, 6))
plt.imshow(phi_2d, extent=[0, L, 0, L], origin='lower', cmap='viridis')
plt.colorbar(label='Solution Value (φ)')
plt.title('Heatmap of the Solution')
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.grid(False)
plt.show()
        