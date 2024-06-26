import roboticstoolbox as rtb
import numpy as np
import spatialmath as sm
from spatialmath import SE3


robot = rtb.models.Panda()


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
E13 = rtb.ET.Rx(np.pi)
E14 = rtb.ET.tz(0.107)
E15 = rtb.ET.Rz()

panda = E1 * E2 * E3 * E4 * E5 * E6 * E7 * E8 * E9 * E10 * E11 * E12 * E13 * E14 * E15
print("Enter the joint angles for panda:")
input_angles=robot.qr
#np.array([])

#for i in range(6):
#    num=float(input("Q{}:".format(i+1)))
#    input_angles.append(num)
q=input_angles*(np.pi/180)
#SOME NP ARRAY TO BE FILLED FOR Q
fk = np.eye(4)

# Now we must loop over the ETs in the Panda
for et in panda:
    if et.isjoint:
        # This ET is a variable joint
        # Use the q array to specify the joint angle for the variable ET
        fk = fk @ et.A(q[et.jindex])
    else:
        # This ET is static
        fk = fk @ et.A()
print(sm.SE3(fk))
print(robot.fkine(robot.qr))