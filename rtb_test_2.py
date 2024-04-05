# We will do the imports required for this notebook here

# numpy provides import array and linear algebra utilities
import numpy as np

# the robotics toolbox provides robotics specific functionality
import roboticstoolbox as rtb

# spatial math provides objects for representing transformations
import spatialmath as sm
import spatialmath.base as smb

# use timeit to benchmark some methods
from timeit import default_timer as timer

# ansitable is a great package for printing tables in a terminal
from ansitable import ANSITable


# We can program these in Python
# These methods assume the inputs are correctly sized and numpy arrays

def vex(mat):
    '''
    Converts a 3x3 skew symmetric matric to a 3 vector
    '''

    return np.array([mat[2, 1], mat[0, 2], mat[1, 0]])

def skew(vec):
    '''
    Converts a 3 vector to a 3x3 skew symmetric matrix
    '''

    return np.array([
        [0, -vec[2], vec[1]],
        [vec[2], 0, -vec[0]],
        [-vec[1], vec[0], 0]
    ])


def vexa(mat):
    '''
    Converts a 4x4 augmented skew symmetric matric to a 6 vector
    '''

    return np.array([mat[0, 3], mat[1, 3], mat[2, 3], mat[2, 1], mat[0, 2], mat[1, 0]])

def skewa(vec):
    '''
    Converts a 6 vector to a 4x4 augmented skew symmetric matrix
    '''

    return np.array([
        [0, -vec[5], vec[4], vec[0]],
        [vec[5], 0, -vec[3], vec[1]],
        [-vec[4], vec[3], 0, vec[2]],
        [0, 0, 0, 0]
    ])

def ρ(tf):
    '''
    The method extracts the rotational component from an SE3
    '''
    return tf[:3, :3]

def τ(tf):
    '''
    The method extracts the translation component from an SE3
    '''
    return tf[:3, 3]

# We can make Python functions which perform these derivatives

def dTRx(θ, flip):
    '''
    Calculates the derivative of an SE3 which is a pure rotation around
    the x-axis by amount θ
    '''
    # This is the [Rhat_x] matrix in the maths above
    Rhx = np.array([
        [0, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0]
    ])

    # This is the T_R_x matrix in the maths above
    Trx = smb.trotx(θ)

    if flip:
        return Rhx.T @ Trx.T
    else:
        return Rhx @ Trx

def dTRy(θ, flip):
    '''
    Calculates the derivative of an SE3 which is a pure rotation around
    the y-axis by amount θ
    '''
    # This is the [Rhat_y] matrix in the maths above
    Rhy = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, 0]
    ])

    # This is the T_R_y matrix in the maths above
    Try = smb.troty(θ)

    if flip:
        return Rhy.T @ Try.T
    else:
        return Rhy @ Try

def dTRz(θ, flip):
    '''
    Calculates the derivative of an SE3 which is a pure rotation around
    the z-axis by amount θ
    '''
    # This is the [Rhat_z] matrix in the maths above
    Rhz = np.array([
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])

    # This is the T_R_z matrix in the maths above
    Trz = smb.trotz(θ)

    if flip:
        return Rhz.T @ Trz.T
    else:
        return Rhz @ Trz

# We can make Python functions which perform these derivatives
# Since these are constant, we don't need to handle joints being
# flipped like we do with revolute joints

def dTtx():
    '''
    Calculates the derivative of an SE3 which is a pure translation along
    the x-axis by amount d
    '''
    # This is the [That_x] matrix in the maths above
    Thx = np.array([
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])

    return Thx

def dTty():
    '''
    Calculates the derivative of an SE3 which is a pure translation along
    the y-axis by amount d
    '''
    # This is the [That_y] matrix in the maths above
    Thy = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])

    return Thy

def dTtz():
    '''
    Calculates the derivative of an SE3 which is a pure translation along
    the z-axis by amount d
    '''
    # This is the [That_z] matrix in the maths above
    Thz = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])

    return Thz

# Before going any further, lets remake the panda model we made in the first notebook
E1 = rtb.ET.tz(0.333) 
E2 = rtb.ET.Rz()
E3 = rtb.ET.Ry() 
E4 = rtb.ET.tz(0.316) 
E5 = rtb.ET.Rz()
E6 = rtb.ET.tx(0.0825) 
E7 = rtb.ET.Ry(flip=True) 
E8 = rtb.ET.tx(-0.0825) 
E9 = rtb.ET.tz(0.384) 
E10 = rtb.ET.Rz()
E11 = rtb.ET.Ry(flip=True)
E12 = rtb.ET.tx(0.088) 
E13 = rtb.ET.Rx(np.pi / 2) 
E14 = rtb.ET.tz(0.107)
E15 = rtb.ET.Rz()
panda = E1 * E2 * E3 * E4 * E5 * E6 * E7 * E8 * E9 * E10 * E11 * E12 * E13 * E14 * E15

# And make a joint coordinate array q
q = np.array([0, -0.3, 0, -2.2, 0, 2, 0.79])


def dET(et, η):
    '''
    This method takes an ET and returns the derivative with respect to the joint coordinate
    This is here for convenience
    '''

    if et.axis == 'Rx':
        return dTRx(η, et.isflip)
    elif et.axis == 'Ry':
        return dTRy(η, et.isflip)
    elif et.axis == 'Rz':
        return dTRz(η, et.isflip)
    elif et.axis == 'tx':
        return dTtx()
    elif et.axis == 'ty':
        return dTty()
    else:
        return dTtz()

def dT_j(ets, j, q):
    '''
    This methods calculates the dervative of an SE3 with respect to joint coordinate j
    '''

    # Allocate an identity matrix for the result
    dT = np.eye(4)

    # Find the jth variable et in the ets
    et_j = ets.joints()[j]

    # Loop overs the ETs in the robot ETS
    for et in ets:
        if et == et_j:
            # This ET is variable and corresponds to joint j
            dT = dT @ dET(et, q[et.jindex])
        elif et.isjoint:
            # This ET is a variable joint
            # Use the q array to specify the joint angle for the variable ET
            dT = dT @ et.A(q[et.jindex])
        else:
            # This ET is static
            dT = dT @ et.A()

    return dT

def Jω_j(ets, j, q):
    '''
    This method calculates the rotational component of the jth column of the manipulator Jacobian
    '''

    # Calculate the forward kinematics at q
    T = ets.eval(q)

    # Calculate the derivative of T with respect to q_j
    dT = dT_j(ets, j, q)

    Jω = vex(ρ(dT) @ (ρ(T).T))

    return Jω
    
def Jv_j(ets, j, q):
    '''
    This method calculates the tranlation component of the jth column of the manipulator Jacobian
    '''

    # Calculate the derivative of T with respect to q_j
    dT = dT_j(ets, j, q)

    Jv = τ(dT)

    return Jv

# By using our previously defined methods, we can now calculate the manipulator Jacobian

def jacob0(ets, q):
    '''
    This method calculates the manipulator Jacobian in the world frame
    '''

    # Allocate array for the Jacobian
    # It is a 6xn matrix
    J = np.zeros((6, ets.n))

    for j in range(ets.n):
        Jv = Jv_j(ets, j, q)
        Jω = Jω_j(ets, j, q)

        J[:3, j] = Jv
        J[3:, j] = Jω

    return J

# Calculate the manipulator Jacobian of the Panda in the world frame
J = jacob0(panda, q)

print(f"The manipulator Jacobian (world-frame) \nin configuration q = {q} \nis: \n{np.round(J, 2)}")
