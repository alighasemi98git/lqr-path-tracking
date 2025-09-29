import numpy as np
import control as ctrl
import matplotlib.pyplot as plt
import control


class LinearCarModel:
    
    def __init__(self, Lf          : float = 1.27, 
                       Lr          : float = 1.90, 
                       mass        : float = 1823, 
                       Iz          : float = 6282,
                       Cf          : float = 42000,
                       Cr          : float = 62000,
                       k_v         : float = 0.1,
                       max_acc     : float = 3.0, 
                       velocity_ref: float = 20.0,
                       ds          : float = 0.1):

        """
        Linear vehicle model for a car.

        :param Lf: Distance from the center of gravity to the front axle (m)
        :type Lf: float
        :param Lr: Distance from the center of gravity to the rear axle (m)
        :type Lr: float
        :param mass: Mass of the vehicle (kg)
        :type mass: float
        :param Iz: Yaw moment of inertia (kg*m^2)
        :type Iz: float
        :param Cf: Cornering stiffness of the front tires (N/rad)
        :type Cf: float
        :param Cr: Cornering stiffness of the rear tires (N/rad)
        :type Cr: float
        :param k_v: Velocity gain (s) (basically determines your look ahead distance)
        :type k_v: float
        :param max_acc: Maximum acceleration (m/s^2)
        :type max_acc: float
        :param velocity_ref: Reference velocity for the vehicle (m/s)
        :type velocity_ref: float float
        :param ds: Discretization time step (s)
        :type ds: float
        """

        self.lf      :float = Lf
        self.lr      :float = Lr
        self.mass    :float = mass
        self.Iz      :float = Iz
        self.Cf      :float = Cf
        self.Cr      :float = Cr
        self.k_v     :float = k_v
        self.lp      :float = self.k_v * velocity_ref  # look ahead distance


        self.max_acc :float = max_acc       # maximum acceleration
        self.l       :float = Lf + Lr       # wheelbase
        self.ds      :float = ds            # discretization time step

        self.k       :np.ndarray = np.array([0.0])           # curvature (0. == straight line)
        self.v_ref   :float = velocity_ref  # reference velocity    
        
        # continuous time dynamics
        self.A_cont  : np.ndarray = None
        self.B_cont  : np.ndarray = None
        self.Bw_cont : np.ndarray = None  # disturbance input matrix
        self.C_cont  : np.ndarray = None
        self.D_cont  : np.ndarray = None
        self.Dw_cont : np.ndarray = None  # disturbance output matrix
        

        # discrete time dynamics
        self.A  : np.ndarray = None
        self.B  : np.ndarray = None
        self.Bw : np.ndarray = None  # disturbance matrix
        self.C  : np.ndarray = None
        self.D  : np.ndarray = None
        self.Dw : np.ndarray = None  # disturbance output matrix

        self.n : int = 5
        self.m : int = 2
        self.p : int = 2


        self._compute_system_matrices(v_ref=velocity_ref)

    

    def _compute_system_matrices(self,v_ref : float = 20.0):
        """
        Compute the system matrices A, B, C, D for the linearized vehicle model.

        :param v_ref: Reference velocity for the vehicle (m/s)
        :type v_ref: float
        """
        
        
        c1 = -(self.Cf + self.Cr)
        c2 = -(self.Cf * self.lf - self.Cr * self.lr)
        c3 = -(self.Cf * self.lf**2 + self.Cr * self.lr**2) 


        # Remove upon completion
        
        
        v  = v_ref
        m  = self.mass
        Iz = self.Iz
        Cf = self.Cf
        lf = self.lf
        Lp = self.lp 


        A = np.array([
            [0.0,        v,      -1.0,                  -Lp,      0.0],              
            [0.0,        0.0,     0.0,                  -1.0,     0.0],              
            [0.0,        0.0,  c1/(m*v),  (c2/(m*v**2) - 1.0),   Cf/m],              
            [0.0,        0.0, c2/(Iz*v),        c3/(Iz*v),    Cf*lf/Iz],             
            [0.0,        0.0,     0.0,                   0.0,     0.0]               
        ])
        
        Bw = np.array([
            [0.0],
            [ v],       
            [0.0],
            [0.0],
            [0.0]
        ])

        B = np.array([
            [0.0,     0.0],
            [0.0,     0.0],
            [0.0,     0.0],
            [0.0,  1.0/Iz],
            [1.0,     0.0]
        ])

        C = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],   
            [0.0, 1.0, 0.0, 0.0, 0.0]    
        ])

        D  = np.zeros((2,2))
        Dw = np.zeros((2,1))


        self.A_cont  = A   * 1/v_ref  
        self.B_cont  = B   * 1/v_ref  
        self.Bw_cont = Bw  * 1/v_ref  
        self.C_cont  = C
        self.D_cont  = D
        self.Dw_cont = Dw
    

    def c2d(self):
        """
        Convert the continuous-time system to a discrete-time system.

        :param Ts: Sampling time (s)
        :type Ts: float
        :return: Discrete-time system matrices A_d, B_d, C_d, D_d
        :rtype: tuple
        """
        
        # create a continuous time system in state space form
        continuous_system = ctrl.ss(self.A_cont, self.B_cont, self.C_cont, self.D_cont)

        continuous_system_disturbance = ctrl.ss(self.A_cont, self.Bw_cont, self.C_cont, self.Dw_cont)
        
        
        # create a discrete time system in state space form
        discrete_system   = ctrl.c2d(continuous_system, self.ds, method='zoh')

        discrete_system_disturbance = ctrl.c2d(continuous_system_disturbance, self.ds, method='zoh')


        # extract the discrete time matrices
        ( Ad_list , Bd_list , Cd_list , Dd_list ) = ctrl.ssdata ( discrete_system  )

        ( _, Bd_list_disturbance , _, Dd_list_disturbance ) = ctrl.ssdata ( discrete_system_disturbance  )
        
        # convret the list to numpy arrays
        self.A  = np.array( Ad_list )
        self.B  = np.array( Bd_list )
        self.C  = np.array( Cd_list )
        self.D  = np.array( Dd_list )
        self.Bw = np.array( Bd_list_disturbance )
        self.Dw = np.array( Dd_list_disturbance )




    def poles_zeros(self, A, B, C, D):
        """
        Plots the system poles and zeros.

        :param A: state transition matrix
        :type A: np.ndarray
        :param B: control matrix
        :type B: np.ndarray
        :param C: state-observation matrix
        :type C: np.ndarray
        :param D: control-observation matrix
        :type D: np.ndarray
        """
        # dt == 0 -> Continuous time system
        # dt != 0 -> Discrete time system
        sys = ctrl.ss(A, B, C, D, dt=self.ds)
        ctrl.pzmap(sys)
        plt.show()
        return
    
    def get_lqr_controller(self,Q,R):
        """
        Solve the discrete-time algebraic Riccati equation (DARE) for the given system.

        :param Q: The state weighting matrix.
        :type Q: np.ndarray
        :param R: The input weighting matrix.
        :type R: np.ndarray

        :returns: Optimal gain of the LQR solution.
        :rtype: np.ndarray
        """

        
        L, _, _ = control.dlqr(self.A, self.B, Q, R)

        return L

        L, _, _ = control.dlqr(self.A, self.B, Q, R)

        return L

