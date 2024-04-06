import roboticstoolbox as rtb
import numpy as np
import spatialmath as sm
from spatialmath import SE3

robot = rtb.models.Panda()
Tep = SE3.Trans(0.6, -0.3, 0.1) * SE3.OA([0, 1, 0], [0, 0, -1])
sol_lm = robot.ik_LM(Tep)
q_final=sol_lm[0]
print(robot.fkine(robot.qr))
print(type(robot.fkine(robot.qr)))
print(robot.fkine(q_final))
qt = rtb.jtraj(robot.qr,q_final , 50)
#robot.plot(qt.q, backend='pyplot', movie='panda1.gif',dt=0.01,limits=[0,1,0,1,0,1],shadow=False,eeframe=True)
