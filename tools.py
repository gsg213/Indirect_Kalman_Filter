import numpy as np
import math
from scipy.linalg import svd

def quaternion2euler(quat):

    euler = np.zeros((3,1))

    q0 = quat[0]
    q1 = quat[1]
    q2 = quat[2]
    q3 = quat[3]
    euler[2] = math.atan2( (2 * q1 * q2 + 2 * q0 * q3) , (2 * q0 * q0 + 2 * q1 * q1 - 1))
    euler[1] = math.asin(-2*q1*q3 + 2*q0*q2)
    euler[0] = math.atan2( (2 * q2 * q3 + 2 * q0 * q1) , (2 * q0 * q0 + 2 * q3 * q3 - 1))

    return euler


def quaternionmul(p,q):
    
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]

    M = np.vstack((np.hstack((p0 , -p1 , -p2 , -p3)), 
                   np.hstack((p1 , p0 , -p3 , p2 )),
                   np.hstack((p2 , p3 , p0, -p1)),
                   np.hstack((p3 , -p2, p1 , p0))))

    v = np.vstack((q[0] , q[1] , q[2]  ,q[3]))

    r = M * v

    return r

def dcm2quaternion(C):
    f = np.zeros((4,1))

    f[0] = 0.25 * (1 + np.trace(C))
    f[1] = 0.25 * (C[0,0] - C[1,1] - C[2,2] + 1)
    f[2] = 0.25 * (C[1,1] - C[0,0] - C[2,2] + 1)
    f[3] = 0.25 * (C[2,2] - C[1,1] - C[0,0] + 1)

    maxf = max(f)
    index = np.where(f == maxf)

    q = np.zeros((4,1))
    if index == 1:
        q[0] = math.sqrt(f[0])
        q[1] = (C[1,2] - C[2,1]) / (4*q[0])
        q[2] = (C[0,2] - C[2,0]) / (-4*q[0])
        q[3] = (C[0,1] - C[1,0]) / (4*q[0])
    elif index == 2 :
        q[1] = math.sqrt(f[1])
        q[0] = (C[1,2] - C[2,1]) / (4*q[1])
        q[2] = (C[0,1] + C[1,0]) / (4*q[1])
        q[3] = (C[0,2] + C[2,0]) / (4*q[1])
    elif  index == 3 :
        q[2] = math.sqrt(f[2])
        q[0] = (C[0,2] - C[2,0]) / (-4*q[2])
        q[1] = (C[0,1] + C[1,0]) / (4*q[2])
        q[3] = (C[1,2] + C[2,1]) / (4*q[2])
    elif index == 4:
        q[3] = math.sqrt(f[3])
        q[0] = (C[0,1] - C[1,0]) / (4*q[3])
        q[1] = (C[0,2] + C[2,0]) / (4*q[3])
        q[2] = (C[1,2] + C[2,1]) / (4*q[3])
    
    return q.reshape(q.shape[0])


def vec2product(v):

    M = np.vstack((np.hstack((0, -v[3], v[2])),
                   np.hstack((v[3], 0, -v[1])),
                   np.hstack((-v[2], v[1], 0))))
    
    return M

def quaternion2dcm(q):

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    M = np.vstack((np.hstack((2*q0^2 + 2*q1^2 - 1, 2*q1*q2 + 2*q0*q3, 2*q1*q3-2*q0*q2)),
                   np.hstack((2*q1*q2 - 2*q0*q3, 2*q0^2 + 2*q2^2 - 1, 2*q2*q3 + 2*q0*q1)),
                   np.hstack((2*q1*q3+2*q0*q2, 2*q2*q3-2*q0*q1, 2*q0^2+2*q3^2-1))
                  ))

    return M
    

def Compute_Attitude(yg,ya,ym,tt,Rg,Ra,Rm):
    """ Attitude estimation using a quaternion-based indirect Kalman filter
    yg: gyroscope output
    ya: accelerometer output
    ym: magnetometer output
    tt: time along data samples
    Rg: noise error covariance matrix for gyroscope
    Ra: noise error covariance matrix for accelerometer
    Rm: noise error covariance matrix for magnetometer  """

    D2R = math.pi/180 # deg to rad

    # Number of data samples: N
    N = max(ya.shape)

    g = 9.8
    g_tild = np.array([[0],[0],[g]])

    alpha = 50*D2R
    m_tild =  np.vstack((math.cos(alpha),
                       0, 
                       -math.sin(alpha)))

    Q_b_g = 0.000001*np.eye(3)
    Q_b_a = 0.000001*np.eye(3)

    Q = np.vstack((np.hstack((0.25*Rg, np.zeros((3,6)))),
                   np.hstack((np.zeros((3,3)),Q_b_g,np.zeros((3,3)))),
                   np.hstack((np.zeros((3,6)),Q_b_a))))

    # Kalman Filter

    # q4: quaternion
    q4 = np.zeros((4,N))

    # eulercom4: euler angles
    eulercom4 = np.zeros((3,N))

    # Estimated bias for gyroscope (bghat) and accelerometer (bahat) 
    bghat = np.zeros((3,1))
    bahat = np.zeros((3,1))

    # inital orientation estimation using the TRIAD method
    yabar = ya[:,0] / np.linalg.norm(ya[:,0])
    ymbar = ym[:,0] / np.linalg.norm(ym[:,0])

    foo1 = np.cross(yabar,ymbar) / np.linalg.norm( np.cross(yabar,ymbar) )

    yabar_ = yabar.reshape(yabar.shape[0],1)
    foo1_ = foo1.reshape(foo1.shape[0],1)
    C = np.hstack((-np.cross(yabar,foo1).reshape(foo1.shape[0],1)  , foo1_ , yabar_))
    q4[:,0] = dcm2quaternion(C)

    # Kalman filter state
    x = np.zeros((9,1))

    P = np.vstack((np.hstack((0.01*np.eye(3), np.zeros((3,6)))),
                   np.hstack((np.zeros((3,3)),0.000001*np.eye(3),np.zeros((3,3)))),
                   np.hstack((np.zeros((3,6)),0.000001*np.eye(3)))))

    wx = yg[0,0]
    wy = yg[1,0]
    wz = yg[2,0]
    omega = np.vstack((np.hstack((0,-wx, -wy, -wz)),
                       np.hstack((wx, 0, wz, -wy)),
                       np.hstack((wy, -wz, 0, wx)),
                       np.hstack((wz, wy, -wx, 0))
                       ))

    # variable used in the adaptive algorithm      
    r2count = 100

    # parameter for adaptive algorithm
    M1 = 3
    M2 = 3
    gamma = 0.1

    R = np.zeros((3,3*N));        

    # Kalman filter loop
    for i in range(1,N):
        # Sampling period T
        T = tt[i]-tt[i-1]
        
        yg_b = -(vec2product(yg[:,i-1]-bghat))
        A = np.vstack((np.hstack((yg_b , -0.5*np.eye(3),np.zeros((3,3)))),
                       np.hstack((np.zeros((6,9))))))
        
        phi_k = np.eye(9) + T*A + 0.5*(A^2)*T^2
        
        Q_d = Q*T + 0.5*A*Q + 0.5*Q*np.transpose(A)
        x = phi_k*x
              
        P_p = phi_k*P*np.transpose(phi_k) + Q_d

        omega_ = omega
        wx = yg[0,i]
        wy = yg[1,i]
        wz = yg[2,i]
        w = [[wx],[wy],[wz]]
        omega = np.vstack((np.hstack((0,-wx, -wy, -wz)),
                           np.hstack((wx, 0, wz, -wy)),
                           np.hstack((wy, -wz, 0, wx)),
                           np.hstack((wz, wy, -wx, 0))))
        
        q4[:,i] =  (np.eye(4) + (3/4)*omega*T  - (1/4)*omega_*T \
                    - (1/6)*((np.linalg.norm(w,2))^2)*(T^2)*np.eye(4) \
                    - (1/24)*omega*omega_*T^2 - (1/48)*((np.linalg.norm(w,2))^2)*omega*T^3)*q4[:,i-1]
        
        Cq = quaternion2dcm(q4[:,i])

        # ----------------------------------------------------
        # two step measurement update
        # ----------------------------------------------------
        
        #Ha
        H1 = np.hstack((2*(vec2product(Cq*g_tild)), np.zeros((3,3)), np.eye(3)))
        
        #Za
        z1 = ya[:,i-1] - Cq*g_tild

        # adaptive algorithm
        fooR1 = (z1 - H1*x) * np.transpose(z1 - H1*x)
        R[:,3*(i-1)+1:3*i] = fooR1
        uk = fooR1
        for j in range(i-1,min([i-(M1-1),1])):
            uk = uk + R[:,3*(j-1)+1:3*j]
        
        uk = uk / M1    
        
        fooR2 = H1*P*np.transpose(H1) + Ra

        u,s,v = svd(uk)
        u1 = u[:,0]
        u2 = u[:,1]
        u3 = u[:,2]

        lambda_ = [s[0], s[1], s[2]]

        mu =  np.hstack(( np.transpose(u1) * fooR2 * u1 , 
                          np.transpose(u2) * fooR2 * u2 , 
                          np.transpose(u3) * fooR2 * u3))

        if max(lambda_ - mu) > gamma :
            r2count = 0
            Qstar = max(lambda_[0]-mu[0],0)*u1*np.transpose(u1) \
                + max(lambda_[1] -mu[1],0)*u2*np.transpose(u2)\
                + max(lambda_[2]-mu[2],0)*u3*np.transpose(u3)
        else:
            r2count = r2count + 1
            if  r2count < M2 :
                Qstar = max(lambda_[0]-mu[0],0)*u1*np.transpose(u1)\
                 + max(lambda_[1]-mu[1],0)*u2*np.transpose(u2) \
                 + max(lambda_[2]-mu[2],0)*u3*np.transpose(u3)
            else:
                Qstar = np.zeros((3,3))
      
        K_a = P_p*np.transpose(H1)/(H1*P_p*np.transpose(H1) + Ra + Qstar)        

        xhat_a =  x + K_a*(z1 - H1*x)        

        P_a = (np.eye((9,9)) - K_a*H1)*P_p*np.transpose(np.eye((9,9))- K_a*H1)\
               + K_a*(Ra + Qstar)*np.transpose(K_a)

        
        qe = np.vstack((1,xhat_a[0:2]))
        
        q4[:,i] = quaternionmul(q4[:,i],qe)
        
        q4[:,i] = q4[:,i]/np.linalg.norm(q4[:,i])
        
        xhat_a[0:2] = np.zeros((3,1))

        Cq = quaternion2dcm(q4[:,i])
        
        H_m = [2*(vec2product(Cq*m_tild)), np.zeros((3,3)),np.zeros((3,3))]
        
        z_m = ym[:,i-1] - Cq*m_tild
        
        P_m =  np.vstack((np.hstack((P_a[0:2,0.2], np.zeros((3,6)))),
                         np.hstack((np.zeros((6,3)), np.zeros((6,6))))
                         ))
        
        r3 = Cq * np.vstack((0,0,1))


        K_m = np.vstack((np.hstack((r3*np.transpose(r3), np.zeros((3,6)))), 
                        np.hstack((np.zeros((6,3)), np.zeros((6,6))))\
              *P_m*np.transpose(H_m)/(H_m*P_m*np.transpose(H_m)+Rm)))

        x = xhat_a + K_m*(z_m-H_m*xhat_a)
        

        P = (np.eye((9,9))-K_m*H_m)*P_a*np.transpose(np.eye((9,9))-K_m*H_m)\
            +K_m*Rm*np.transpose(K_m)
        
        bghat = bghat + x[3:5]
        x[3:5] = np.zeros((3,1))

        bahat = bahat + x[6:8]
        x[6:8] = np.zeros((3,1))

        qe = np.vstack((1,xhat_a[0:2]))
        
        q4[:,i] = quaternionmul(q4[:,i],qe)
        
        q4[:,i] = q4[:,i]/np.linalg.norm(q4[:,i])
        
        Cq = quaternion2dcm(q4[:,i])
        
        eulercom4[:,i]=quaternion2euler(q4[:,i])

    return q4, eulercom4, bahat, bghat

   
