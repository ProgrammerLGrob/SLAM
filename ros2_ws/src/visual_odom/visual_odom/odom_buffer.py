from collections import deque
from dataclasses import dataclass
from nav_msgs.msg import Odometry
from visual_odom.constants import State
from scipy.spatial.transform import Rotation



@dataclass
class odom_message:
    stamp: float
    pose: State
    msg: Odometry

class OdomBuffer:
    def __init__(self, maxlen=1000):
        self.Odom_Buffer = deque(maxlen=maxlen)

    def add_odom_message(self, msg: Odometry):
        #if not isinstance(msg, Odometry):
        #    raise TypeError("Only Odometry allowed")

        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        rot = Rotation.from_quat([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        odom_state = State(msg.pose.pose.position.x, msg.pose.pose.position.y, rot.as_euler('xyz')[2])
        data = odom_message(stamp, odom_state ,msg)
        self.Odom_Buffer.append(data)
    def get_latest(self):
        return self.Odom_Buffer[-1]
    def get_interpolated(self, stamp: float):
        buffer = self.Odom_Buffer

        if len(buffer) < 2:                 #return if buffer < 2
            return None

        for i in range(len(buffer) - 1):    #iterate to find fitting stamp
            msg_before = buffer[i]
            msg_after = buffer[i + 1]

            if msg_before.stamp <= stamp <= msg_after.stamp:
                ratio = (stamp - msg_before.stamp) / (msg_after.stamp - msg_before.stamp)

                x = msg_before.pose.x + ratio * (msg_after.pose.x - msg_before.pose.x)
                y = msg_before.pose.y + ratio * (msg_after.pose.y - msg_before.pose.y)

                return State(x, y, 0.0)

        return None